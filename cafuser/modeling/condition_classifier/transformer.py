import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from oneformer.modeling.transformer_decoder.oneformer_transformer_decoder import MLP
from oneformer.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from detectron2.utils.registry import Registry
from detectron2.config import configurable
from detectron2.layers import ShapeSpec

CONDITION_CLASSIFIER_REGISTRY = Registry("CONDITION_CLASSIFIER")
CONDITION_CLASSIFIER_REGISTRY.__doc__ = """
Registry for condition classifier modules for CAFuser.
"""

def build_condition_classifier(cfg, in_channels, in_feature_level):
    name = cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.NAME
    return CONDITION_CLASSIFIER_REGISTRY.get(name)(cfg, in_channels, in_feature_level)

@CONDITION_CLASSIFIER_REGISTRY.register()
class TransformerConditionClassifier(nn.Module):
    @configurable
    def __init__(self, input_dim, in_feature_level, in_modality, hidden_dim, output_dim, 
    nheads, dim_feedforward, enc_layers, dec_layers, dropout, pre_norm):
        super(TransformerConditionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.in_feature_level = in_feature_level
        self.in_modality = in_modality
        self.enc_layers = enc_layers

        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
                
        # Positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            norm_first=pre_norm,
        )

        # Query embeddings for transformer decoder
        self.query_embed = nn.Embedding(output_dim, hidden_dim)        

        self.norm = nn.LayerNorm(hidden_dim)

    @classmethod
    def from_config(cls, cfg, in_channels, in_feature_level):
        if 'MAIN' == cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.IN_MODALITY:
            in_modality = cfg.DATASETS.MODALITIES.MAIN_MODALITY.upper()
        else:
            in_modality = cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.IN_MODALITY
        output_dim = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.N_CTX + len(cfg.MODEL.CONDITION_CLASSIFIER.CONDITION_TEXT_ENTRIES)
        return {
            "input_dim": in_channels[in_feature_level].channels,
            "in_feature_level": in_feature_level,
            "in_modality": in_modality,
            "hidden_dim": cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.HIDDEN_DIM,
            "output_dim": output_dim,
            "nheads": cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.NHEADS,
            "dim_feedforward": cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.DIM_FEEDFORWARD,
            "enc_layers": cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.ENC_LAYERS,
            "dec_layers": cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.DEC_LAYERS,
            "dropout": cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.DROPOUT,
            "pre_norm": cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.PRE_NORM,
        }

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def output_shape(self):
        return ShapeSpec(
            channels=self.transformer.decoder.layers[-1].linear2.out_features,
            height=None,
            width=None,
            stride=None
        )

    def forward(self, feats):
        src = self.input_proj(feats[self.in_feature_level][self.in_modality])  # Shape: [batch_size, hidden_dim, H, W]
        pos_embed = self.pe_layer(src)  # Shape: [batch_size, hidden_dim, H, W]
  
        # Flatten the input features and add positional embeddings
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # Shape: [H*W, batch_size, hidden_dim]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # Shape: [H*W, batch_size, hidden_dim]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # Shape: [num_queries, batch_size, hidden_dim]

        # Apply positional embedding
        src = self.with_pos_embed(src, pos_embed)

        x = self.transformer(src, query_embed) 

        # Normalize the output of the transformer
        x = self.norm(x)        
        x = x.permute(1, 0, 2)

        return x
