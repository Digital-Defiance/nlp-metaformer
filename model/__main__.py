
import torch.nn as nn
import tiktoken
from pydantic_settings import BaseSettings

from core.constants import DEVICE
from core.mixins import MyBaseSettingsMixin
from model.sequence_encoder import SequenceEncoder
from model.transformer_block import TransformerBlock, TransformerJunction
from typing import Literal
from pydantic import model_validator
from core.types import PositiveInt, TensorFloat, TensorInt

gpt2_encoder = tiktoken.get_encoding("gpt2")



class EncoderDecoder(nn.Module):
    def __init__(self, params: "ModelFactory"):
        super(EncoderDecoder, self).__init__()

        self.sequence_encoder = SequenceEncoder(params)

        self.encoder = nn.Sequential()
        for i in range(self.number_of_blocks):
            block = TransformerBlock(params, is_decoder = False)
            self.encoder.add_module(f"encoder_block_{i}", block)

        decoder_blocks = nn.ModuleList()
        junction_blocks = nn.ModuleList()
        for i in range(self.number_of_blocks):
            decoder_blocks.add_module(f"decoder_block_{i}", TransformerBlock(params, is_decoder = True))
            junction_blocks.add_module(f"junction_block_{i}", TransformerJunction(params))

        self.output_layer = nn.Sequential(
            nn.LayerNorm(params.coordinates),
            nn.Linear(params.coordinates, params.tokens, bias=params.bias)
        )

    
    def forward(self, sequence_bw: TensorInt) -> TensorFloat:
        sequence_bwc = self.sequence_encoder(sequence_bw)
        encoder_output_bwc = self.encoder(sequence_bwc)

        for i in range(self.number_of_blocks):
            sequence_bwc = self.decoder_blocks[i](sequence_bwc)
            sequence_bwc = self.junction_blocks[i](sequence_bwc, encoder_output_bwc)

        return self.output_layer(sequence_bwc)


        


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
        return self.create_decoder_only_network()
    
    def create_decoder_only_network(self) -> nn.Module:
        model = nn.Sequential()
        model.add_module("sequence_encoder", SequenceEncoder(self))
        for i in range(self.number_of_blocks):
            model.add_module(f"block_{i}", TransformerBlock(self, is_decoder=True))
        model.add_module("layer_norm_c", nn.LayerNorm(self.coordinates))
        model.add_module("language_model_weights_tc", nn.Linear(self.coordinates, self.tokens, bias=self.bias))
        return model.to(DEVICE)

    def create_encoder_only_network(self) -> nn.Module:
        model = nn.Sequential()
        model.add_module("sequence_encoder", SequenceEncoder(self))
        for i in range(self.number_of_blocks):
            model.add_module(f"block_{i}", TransformerBlock(self, is_decoder = False))
        model.add_module("layer_norm_c", nn.LayerNorm(self.coordinates))
        model.add_module("language_model_weights_tc", nn.Linear(self.coordinates, self.tokens, bias=self.bias))
        return model.to(DEVICE)

    def create_encoder_decoder(self):
        return EncoderDecoder(self).to(DEVICE)

    @classmethod
    def create_variant(cls, variant: str = "NanoGPT") -> nn.Module:
        assert variant in  ["NanoGPT", "NanoMTN"] , f"Unknown variant {variant}"

        if variant == "NanoGPT":
            return cls(
                words=100,
                coordinates=300,
                number_of_blocks=3,
                number_of_heads=3,
                bias = False,
                attention="scaled_dot_product"
            )

        if variant == "NanoMTN":
            return cls(
                words=100,
                coordinates=300,
                number_of_blocks=3,
                number_of_heads=3,
                bias = False,
                attention="metric"
            )
        



