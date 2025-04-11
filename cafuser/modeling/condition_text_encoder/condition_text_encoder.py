import torch
import torch.nn as nn
from detectron2.config.config import CfgNode
from detectron2.config import configurable
from oneformer.modeling.transformer_decoder.text_transformer import TextTransformer
from oneformer.modeling.transformer_decoder.oneformer_transformer_decoder import MLP
from einops import rearrange
from oneformer.data.tokenizer import SimpleTokenizer, Tokenize
from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

CONDITION_TEXT_ENCODER_REGISTRY = Registry("CONDITION_TEXT_ENCODER")
CONDITION_TEXT_ENCODER_REGISTRY.__doc__ = """
Registry for condition text encoder modules for CAFuser.
"""

def build_condition_text_encoder(cfg, in_channels):
    name = cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.NAME
    if 'NAME' in cfg.MODEL.CONDITION_CLASSIFIER.keys(): 
        assert name == cfg.MODEL.CONDITION_CLASSIFIER.NAME, f'Missmatch between {name} and {cfg.MODEL.CONDITION_CLASSIFIER.NAME}: CONDITION_CLASSIFIER.NAME is depreciated, use TEXT_ENCODER.NAME instead'
    return CONDITION_TEXT_ENCODER_REGISTRY.get(name)(cfg)

@CONDITION_TEXT_ENCODER_REGISTRY.register()
class ConditionTextEncoder(nn.Module):
    @configurable
    def __init__(self, 
                 text_encoder: nn.Module,
                 text_projector: nn.Module, 
                 prompt_ctx: nn.Module,
                 text_tokenizer: nn.Module,
                 device: torch.device):
        super(ConditionTextEncoder, self).__init__()
        self.text_encoder = text_encoder
        self.text_projector = text_projector
        self.prompt_ctx = prompt_ctx        
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.to(device)

    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.IS_TRAIN:
            text_encoder = TextTransformer(context_length=cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.CONTEXT_LENGTH,
                                           width=cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.WIDTH,
                                           layers=cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.NUM_LAYERS,
                                           vocab_size=cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.VOCAB_SIZE)
            text_projector = MLP(text_encoder.width, cfg.MODEL.ONE_FORMER.HIDDEN_DIM, 
                                 cfg.MODEL.ONE_FORMER.HIDDEN_DIM, cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.PROJ_NUM_LAYERS)
            if cfg.MODEL.TEXT_ENCODER.N_CTX > 0:
                prompt_ctx = nn.Embedding(cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.N_CTX, cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.WIDTH)
            else:
                prompt_ctx = None
        else:
            text_encoder = None
            text_projector = None
            prompt_ctx = None

        text_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=cfg.MODEL.CONDITION_CLASSIFIER.CONDITION_SEQ_LEN)
        device = torch.device(cfg.MODEL.DEVICE if cfg.MODEL.DEVICE else 'cuda' if torch.cuda.is_available() else 'cpu')

        return {
            "text_encoder": text_encoder,
            "text_projector": text_projector,
            "prompt_ctx": prompt_ctx,
            "text_tokenizer": text_tokenizer,
            "device": device,
        }

    def encode_text(self, text):
        assert text.ndim in [2, 3], text.ndim
        b = text.shape[0]
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True

        # [B, C]
        x = self.text_encoder(text)

        text_x = self.text_projector(x)

        if squeeze_dim:
            text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            if self.prompt_ctx is not None:
                text_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_x.shape[0], 1, 1)
                text_x = torch.cat([text_x, text_ctx], dim=1)
        
        return {"condition_texts": text_x}

    def output_shape(self):
        return ShapeSpec(
            channels=self.text_projector.layers[-1].out_features,
            height=None,
            width=None,
            stride=None
        )

    def forward(self, batched_inputs):
        assert self.training, 'module only available in training mode'
        texts = torch.cat([self.text_tokenizer(x["condition_text"]).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0)
        condition_texts = self.encode_text(texts)
        return condition_texts
