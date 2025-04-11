# ------------------------------------------------------------------------------
# Reference: https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from detectron2.config.config import CfgNode
from detectron2.config import configurable
from detectron2.utils.registry import Registry

FEATURE_ADAPTER_REGISTRY = Registry("FEATURE_ADAPTER")
FEATURE_ADAPTER_REGISTRY.__doc__ = """
Registry for feature adapter modules for CAFuser.
"""

def build_feature_adapter(cfg, in_channels, modalities):
    name = cfg.MODEL.FEATURE_ADAPTER.NAME
    return FEATURE_ADAPTER_REGISTRY.get(name)(cfg, in_channels, modalities)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


@FEATURE_ADAPTER_REGISTRY.register()
class MLPAdapter(nn.Module):
    @configurable
    def __init__(self, input_shapes, levels, modalities, ratio, reduction):
        super(MLPAdapter, self).__init__()
        self.levels = levels
        self.ratio = ratio
        
        # Initialize adapters for each specified level and modality
        self.adapter = nn.ModuleDict({
            level: nn.ModuleDict({
                modality: Adapter(input_shapes[level].channels, reduction[modality])
                for modality in modalities
            })
            for level in levels
        })

    @classmethod
    def from_config(cls, cfg, input_shapes, modalities):
        ratio = {modality: cfg.DATASETS.MODALITIES[modality].FEATURE_ADAPTER.RATIO for modality in modalities}
        reduction = {modality: cfg.DATASETS.MODALITIES[modality].FEATURE_ADAPTER.REDUCTION for modality in modalities}
        return {
            "input_shapes": input_shapes,
            "levels": cfg.MODEL.FUSION.LEVELS,
            "modalities": modalities,
            "ratio": ratio,
            "reduction": reduction,
        }

    def forward(self, src):
        for level, features in src.items():
            if level in self.levels:
                for modality, feature in features.items():
                    BS, C, H, W = feature.size()
                    x = feature.view(BS, C, -1).permute(0, 2, 1).contiguous()  # shape: (batch_size, height * width, channels)
                    y = self.adapter[level][modality](x)
                    x = self.ratio[modality] * y + (1 - self.ratio[modality]) * x
                    x = x.permute(0, 2, 1).contiguous().view(BS, C, H, W)  # back to original shape
                    features[modality] = x
        return src
