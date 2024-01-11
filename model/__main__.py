
import torch.nn as nn
import tiktoken
from pydantic_settings import BaseSettings

from core.constants import DEVICE
from core.mixins import MyBaseSettingsMixin
from model.sequence_encoder import SequenceEncoder
from model.transformer_block import TransformerBlock
from typing import Literal
from pydantic import model_validator
from core.types import PositiveInt

gpt2_encoder = tiktoken.get_encoding("gpt2")


class ModelFactory(BaseSettings, MyBaseSettingsMixin):

    coordinates: PositiveInt = 400
    tokens: PositiveInt = gpt2_encoder.max_token_value
    words: PositiveInt = 1000
    number_of_blocks: PositiveInt = 10
    number_of_heads: PositiveInt = 20
    bias: bool = False
    attention: Literal["metric", "scaled_dot_product"] = "scaled_dot_product"

    class Config:
        env_prefix = "MODEL_"

    @model_validator(mode='after')
    def validate(self) -> 'ModelFactory':
        assert self.coordinates % self.number_of_heads == 0, "Coordinates must be divisible by number of heads"
        return self

    def estimate_model_size(self) -> int:
        # calculate a rough estimate of the total number of parameters in the model
        n_sequence_encoder = self.coordinates * self.words + self.coordinates * self.tokens
        n_transformer_blocks = self.coordinates * self.coordinates * 4 * self.number_of_blocks
        n_layer_norm_c = self.coordinates * 2
        n_language_model_weights_tc = self.coordinates * self.tokens
        n_total = n_sequence_encoder + n_transformer_blocks + n_layer_norm_c + n_language_model_weights_tc
        return n_total
        # occupied_memory_gb = n_total * 4 / 1024 / 1024 / 1024
        # return self


    def create_model(self) -> nn.Module:
        model = nn.Sequential()
        model.add_module("sequence_encoder", SequenceEncoder(self))
    
        for i in range(self.number_of_blocks):
            model.add_module(f"block_{i}", TransformerBlock(self))

        model.add_module("layer_norm_c", nn.LayerNorm(self.coordinates))
        model.add_module("language_model_weights_tc", nn.Linear(self.coordinates, self.tokens, bias=self.bias))
        return model.to(DEVICE)


    

