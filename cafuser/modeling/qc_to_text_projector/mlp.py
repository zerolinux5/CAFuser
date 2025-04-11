import torch
import torch.nn as nn
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

QC_TO_TEXT_PROJECTOR_REGISTRY = Registry("QC_TO_TEXT_PROJECTOR")
QC_TO_TEXT_PROJECTOR_REGISTRY.__doc__ = """
Registry for mapping QC (query condition) to text features for CAFuser.
"""

def build_qc_to_text_projector(cfg, in_channels):
    name = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.QC_TO_TEXT_PROJECTOR.NAME
    return QC_TO_TEXT_PROJECTOR_REGISTRY.get(name)(cfg, in_channels)

@QC_TO_TEXT_PROJECTOR_REGISTRY.register()
class Mlp(nn.Module):
    """Multilayer perceptron."""

    @configurable
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    @classmethod
    def from_config(cls, cfg, in_channels):
        hidden_features = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.QC_TO_TEXT_PROJECTOR.HIDDEN_FEATURES
        out_features = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.QC_TO_TEXT_PROJECTOR.OUT_FEATURES
        drop = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.QC_TO_TEXT_PROJECTOR.DROP

        return {
            "in_features": in_channels.channels,
            "hidden_features": hidden_features,
            "out_features": out_features,
            "drop": drop,
        }

    def output_shape(self):
        return self.fc2.out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x