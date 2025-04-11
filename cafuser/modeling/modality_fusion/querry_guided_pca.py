import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
import math
from .prallel_cross_attention import MODALITY_FUSION_REGISTRY, WindowMCA, ParallelCrossAttention


class MultiWindowCrossAttention(nn.Module):
    def __init__(self,
                 window_size=7,
                 with_pad_mask=False,
                 cat_qc_to_primary_modality=True,
                 cat_qc_to_secondary_modality=True,
                 **kwargs):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.with_pad_mask = with_pad_mask
        self.attn = WindowMCA(window_size=self.window_size, with_rpe=True, with_qc=True, **kwargs)
        self.cat_qc_to_primary_modality = cat_qc_to_primary_modality
        self.cat_qc_to_secondary_modality = cat_qc_to_secondary_modality
        assert self.cat_qc_to_primary_modality or self.cat_qc_to_secondary_modality, 'If no querry guidance is wanted use ParallelCrossAttention instead'

    def forward(self, x, y, H, W, qc, **kwargs):
        assert x.shape == y.shape
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        Wh, Ww = self.window_size

        # center-pad the feature on H and W axes
        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = F.pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        y = F.pad(y, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

        # permute
        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww, C)
        y = y.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww, C)

        # concatanate with q_condition:
        qc_expanded = qc.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, math.ceil(H / Wh), 1, math.ceil(W / Ww), 1, C).reshape(-1, 1, C)
        if self.cat_qc_to_primary_modality:
            x = torch.cat((x, qc_expanded), dim=1)
        if self.cat_qc_to_secondary_modality:
            y = torch.cat((y, qc_expanded), dim=1)

        # attention
        if self.with_pad_mask and pad_h > 0 and pad_w > 0:
            pad_mask = x.new_zeros(1, H, W, 1)
            pad_mask = F.pad(pad_mask, [0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=-float('inf'))
            pad_mask = pad_mask.view(1, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww)
            pad_mask = pad_mask[:, None, :].expand([-1, Wh * Ww, -1])
        
            # add 0 to qc, so that we still pay attention to it
            if self.cat_qc_to_primary_modality:
                query_mask = x.new_zeros(pad_mask.shape[0], 1, Wh * Ww)
                pad_mask = torch.cat((pad_mask, query_mask), dim=1)
                if self.cat_qc_to_secondary_modality:
                    key_mask = x.new_zeros(pad_mask.shape[0], (Wh * Ww) + 1, 1)
                    pad_mask = torch.cat((pad_mask, key_mask), dim=2)
            elif self.cat_qc_to_secondary_modality:
                key_mask = x.new_zeros(pad_mask.shape[0], Wh * Ww, 1)
                pad_mask = torch.cat((pad_mask, key_mask), dim=2)

            out = self.attn(x, y, y, pad_mask, **kwargs)
        else:
            out = self.attn(x, y, y, **kwargs)

        if self.cat_qc_to_primary_modality:
            # remove q_condition:
            out = out[:,:-1,:]

        # reverse permutation
        out = out.reshape(B, math.ceil(H / Wh), math.ceil(W / Ww), Wh, Ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H + pad_h, W + pad_w, C)
        
        # de-pad
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]
        return out.reshape(B, N, C)

class WeatherGuidedFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, window_size=7, mlp_ratio=4, drop_path=0.0, act_cfg=dict(type='GELU'), norm_cfg=dict(type='SyncBN'), transformer_norm_cfg=dict(type='LN', eps=1e-6), main_modality=["CAMERA"], secondary_modalities=["LIDAR", "EVENT_CAMERA", "RADAR", "REF_IMAGE"], **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.main_modality = main_modality
        self.secondary_modalities = secondary_modalities

        self.norm1 = nn.ModuleDict({modality: nn.LayerNorm(in_channels) for modality in self.secondary_modalities})
        self.norm2 = nn.ModuleDict({modality: nn.LayerNorm(out_channels) for modality in self.secondary_modalities})
        self.attn = nn.ModuleDict({modality: MultiWindowCrossAttention(embed_dim=in_channels, num_heads=num_heads, window_size=window_size, **kwargs) for modality in self.secondary_modalities})
        
        self.norm3 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(in_channels * mlp_ratio, out_channels),
        )
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)

    def forward(self, modality_features, qc):
        B, C, H, W = modality_features[self.main_modality].size()
        x = modality_features[self.main_modality].view(B, -1, C)
        x_tmp = x.clone()
        for modality in self.secondary_modalities:
            z = modality_features[modality].view(B, -1, C)
            x = x + z + self.drop_path(self.attn[modality](self.norm1[modality](x_tmp), self.norm2[modality](z), H, W, qc))
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        x = x.view(B, C, H, W)
        return x


@MODALITY_FUSION_REGISTRY.register()
class WeatherConditionedPCA(ParallelCrossAttention):
    def __init__(self, 
                 cfg,
                 modalities,
                 input_shapes,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 ):
        super().__init__(cfg, modalities, input_shapes, norm_cfg, transformer_norm_cfg)
        self.expects_q_condition = True
        self.project_qc = cfg.MODEL.FUSION.QC.ENABLE_WEATHER_PROJ
        if self.project_qc:
            weather_num_queries = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.N_CTX + \
                len(cfg.MODEL.CONDITION_CLASSIFIER.CONDITION_TEXT_ENTRIES)
            weather_dim = cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.HIDDEN_DIM * \
                                weather_num_queries
            self.weather_proj = nn.ModuleDict({
                level: nn.Linear(weather_dim, input_shapes[level].channels)
                for level in self.levels
                })


    def align_qc_dim(self, level, q_condition):
        if self.project_qc:
            # Project weather query to match the feature dimension
            q_condition_flat = q_condition.flatten(1)
            projected_qc = self.weather_proj[level](q_condition_flat).unsqueeze(1)
            return projected_qc
        else:
            return q_condition

    def forward(self, src, q_condition):
        outputs = {}

        for level, features in src.items():
            if level in self.levels:
                qc = self.align_qc_dim(level, q_condition)
                # Concatenate weather query with modality features
                B, C, H, W = features[self.modalities[0]].shape
                outputs[level] = self.fusion[level](features, qc)
            else:
                outputs[level] = features[self.modalities[0]]
        return outputs

    def _make_multimodal_fusion(self, cfg, layer_config, input_shapes, transformer_norm_cfg, norm_cfg):
        num_heads = layer_config.NHEAD
        num_window_size = layer_config.WINDOW_SIZE
        num_mlp_ratio = layer_config.MLP_RATIO
        drop_path = layer_config.DROP_PATH
        proj_drop_rate = layer_config.PROJ_DROP_RATE
        attn_drop_rate = layer_config.ATTN_DROP_RATE
        with_pad_mask = layer_config.WITH_PAD_MASK
        

        cat_qc_to_primary_modality = cfg.MODEL.FUSION.QC.CAT_QC_TO_QUERY
        if 'CAT_QC_TO_QUERY' in cfg.MODEL.FUSION.PCA.keys():
            # cat_qc_to_primary_modality = cfg.MODEL.FUSION.PCA.CAT_QC_TO_QUERY
            assert cfg.MODEL.FUSION.PCA.CAT_QC_TO_QUERY == cfg.MODEL.FUSION.QC.CAT_QC_TO_QUERY, f'Missmatch: {cfg.MODEL.FUSION.PCA.CAT_QC_TO_QUERY} != {cfg.MODEL.FUSION.QC.CAT_QC_TO_QUERY}. PCA.CAT_QC_TO_QUERY is depreciated, use QC.CAT_QC_TO_QUERY instead'
        cat_qc_to_secondary_modality = cfg.MODEL.FUSION.QC.CAT_QC_TO_KEY
        if 'CAT_QC_TO_KEY' in cfg.MODEL.FUSION.PCA.keys():
            # cat_qc_to_secondary_modality = cfg.MODEL.FUSION.PCA.CAT_QC_TO_KEY
            assert cfg.MODEL.FUSION.PCA.CAT_QC_TO_KEY == cfg.MODEL.FUSION.QC.CAT_QC_TO_KEY, f'Missmatch: {cfg.MODEL.FUSION.PCA.CAT_QC_TO_KEY} != {cfg.MODEL.FUSION.QC.CAT_QC_TO_QUERY}. PCA.CAT_QC_TO_KEY is depreciated, use QC.CAT_QC_TO_KEY instead'\


        fusion_modules = nn.ModuleDict()
        
        for i, level in enumerate(self.levels):
            fusion_modules[level] = WeatherGuidedFusionBlock(
                input_shapes[level].channels,
                input_shapes[level].channels,
                num_heads=num_heads[self.levels_map[level]],
                window_size=num_window_size,
                mlp_ratio=num_mlp_ratio,
                drop_path=drop_path,
                norm_cfg=norm_cfg,
                transformer_norm_cfg=transformer_norm_cfg,
                main_modality=self.modalities[0],
                secondary_modalities=self.modalities[1:],
                proj_drop_rate=proj_drop_rate,
                attn_drop_rate=attn_drop_rate,
                cat_qc_to_primary_modality=cat_qc_to_primary_modality,
                cat_qc_to_secondary_modality=cat_qc_to_secondary_modality,
                with_pad_mask=with_pad_mask,
            )

        return fusion_modules