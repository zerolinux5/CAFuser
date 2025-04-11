import torch
import torch.nn as nn
from detectron2.utils.registry import Registry
from .prallel_cross_attention import build_modality_fusion, MODALITY_FUSION_REGISTRY


@MODALITY_FUSION_REGISTRY.register()
class QueryGuidedAdditionFusion(nn.Module):
    def __init__(self, cfg, modalities, _=None):
        super(QueryGuidedAdditionFusion, self).__init__()
        self.expects_q_condition = True
        self.modalities = modalities
        self.fusion_levels = cfg.MODEL.FUSION.LEVELS
        self.main_modality = modalities[0]
        assert self.main_modality == cfg.DATASETS.MODALITIES.MAIN_MODALITY.upper()

        self.input_dim = cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.HIDDEN_DIM
        self.input_queries = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.N_CTX + len(cfg.MODEL.CONDITION_CLASSIFIER.CONDITION_TEXT_ENTRIES)

        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim * self.input_queries, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, len(modalities)),
            nn.Softmax(dim=-1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of the linear layer to favor the main modality.
        This is done by setting the weights of the main modality to 1 and the rest to 0.
        """
        with torch.no_grad():
            linear_layer = self.ffn[2]
            main_modality_index = self.modalities.index(self.main_modality)
            linear_layer.weight.fill_(0)
            linear_layer.bias.fill_(0)
            linear_layer.bias[main_modality_index] = 1

    def forward(self, features, q_condition):
        """
        Args:
            features: Dictionary containing the features from different modalities.
            q_condition: Tensor containing the query condition. Shape:
                [batch_size, input_queries, input_dim].

        Returns:
            Dictionary containing the fused features for each level.
        """
        fused_features = {}

        q_condition_flat = q_condition.flatten(1)  # shape: [bs, input_queries * input_dim]
        weights = self.ffn(q_condition_flat)

        for level, modality_features in features.items():
            if level in self.fusion_levels:
                stacked_features = torch.stack([modality_features[modality] for modality in self.modalities], dim=1)
                weights_expanded = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(stacked_features)
                fused_feature = (stacked_features * weights_expanded).sum(dim=1)  # shape: [bs, C, H, W]
            else:
                fused_feature = modality_features[self.main_modality]
            
            fused_features[level] = fused_feature
        
        return fused_features