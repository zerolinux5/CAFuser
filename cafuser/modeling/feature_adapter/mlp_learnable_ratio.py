# ------------------------------------------------------------------------------
# Reference: https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from detectron2.config.config import CfgNode
from detectron2.config import configurable

from .mlp import Adapter, FEATURE_ADAPTER_REGISTRY

def logit(p):
    return torch.log(p / (1 - p))

@FEATURE_ADAPTER_REGISTRY.register()
class MLPAdapterWithLearnableRatio(nn.Module):
    @configurable
    def __init__(self, input_shapes, levels, modalities, reduction, initial_ratio=0.4):
        super(MLPAdapterWithLearnableRatio, self).__init__()
        self.levels = levels
        
        # Initialize adapters for each specified level and modality
        self.adapter = nn.ModuleDict({
            level: nn.ModuleDict({
                modality: Adapter(input_shapes[level].channels, reduction[modality])
                for modality in modalities
            })
            for level in levels
        })        

        initial_ratio = torch.tensor(initial_ratio)
        initial_logit = logit(initial_ratio)
        self.ratios = nn.ModuleDict({
            level: nn.ParameterDict({
                modality: nn.Parameter(initial_logit)
                for modality in modalities
            })
            for level in levels
        })

    @classmethod
    def from_config(cls, cfg, input_shapes, modalities):
        if "FEATURE_ADAPTER" in cfg.DATASETS.MODALITIES[modalities[0]].keys():
            reduction = {modality: cfg.DATASETS.MODALITIES[modality].FEATURE_ADAPTER.REDUCTION for modality in modalities}
        else:
            # > V 1.0
            reduction = {modality: cfg.MODEL.FEATURE_ADAPTER.MLP.REDUCTION for modality in modalities}
        return {
            "input_shapes": input_shapes,
            "levels": cfg.MODEL.FUSION.LEVELS,
            "modalities": modalities,
            "reduction": reduction,
        }

    def forward(self, src):
        for level, features in src.items():
            if level in self.levels:
                for modality, feature in features.items():
                    BS, C, H, W = feature.size()
                    x = feature.view(BS, C, -1).permute(0, 2, 1).contiguous()  # shape: (batch_size, height * width, channels)
                    y = self.adapter[level][modality](x)
                    ratio = torch.sigmoid(self.ratios[level][modality])  # Ensure ratio is between 0 and 1
                    
                    x = ratio * y + (1 - ratio) * x
                    x = x.permute(0, 2, 1).contiguous().view(BS, C, H, W)  # back to original shape
                    features[modality] = x

        return src
