
import torch.nn as nn
import tiktoken
from pydantic_settings import BaseSettings

from core.mixins import MLFlowSaveAndLoadMixin
from model.sequence_encoder import SequenceEncoder
from model.transformer_block import TransformerBlock

gpt2_encoder = tiktoken.get_encoding("gpt2")

class ModelFactory(BaseSettings, MLFlowSaveAndLoadMixin):

    coordinates: int = 3*100
    tokens: int = gpt2_encoder.max_token_value
    words: int = 100
    number_of_blocks: int = 10
    number_of_heads: int = 3
    bias: bool = False

    def make_model(self) -> nn.Module:
        model = nn.Sequential()
        model.add_module("sequence_encoder", SequenceEncoder(self))
    
        for i in range(self.number_of_blocks):
            model.add_module(f"block_{i}", TransformerBlock(self))

        model.add_module("layer_norm_c", nn.LayerNorm(self.coordinates))
        model.add_module("language_model_weights_tc", nn.Linear(self.coordinates, self.tokens, bias=self.bias))
        return model

