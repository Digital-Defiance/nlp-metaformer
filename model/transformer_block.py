
import torch.nn as nn
from typing import Protocol, Literal
from core.types import TensorFloat
from model.self_attention import MetricSelfAttention, ScaledDotProductAttention
from core.logger import get_logger

logger = get_logger(__name__)

class TransformerBlockParameters(Protocol):
    coordinates: int
    words: int
    tokens: int
    attention: Literal["metric", "scaled_dot_product"]


def make_perceptron_layer(coordinates: int) -> nn.Sequential:
    """ Make a perceptron layer. """

    return nn.Sequential(
        nn.LayerNorm(coordinates),
        nn.Linear(coordinates, coordinates // 4),
        nn.Dropout(0.1),
        nn.GELU(),
        nn.Linear(coordinates // 4, coordinates),
        nn.Dropout(0.1),
    )

def make_attention_layer(params, is_causal: bool) -> nn.Sequential:
    """ Make an attention layer. """
    return  MetricSelfAttention(params, is_causal) if params.attention == "metric" else ScaledDotProductAttention(params, is_causal),

class TransformerJunctionBlock(nn.Module):

    def __init__(self, params: TransformerBlockParameters):
        super(TransformerJunctionBlock, self).__init__()
        self.attention_layer_1 = make_attention_layer(params, is_causal=True)
        self.perceptron_layer_1 = make_perceptron_layer(params)
        self.attention_layer_2 = make_attention_layer(params, is_causal=True)
        self.perceptron_layer_2 = make_perceptron_layer(params)

    def forward(self, sequence_bwc: TensorFloat, encoder_output_bwc: TensorFloat) -> TensorFloat:
        
        # Perform self attention with the decoder sequence only
        sequence_bwc = sequence_bwc + self.attention_layer_1(sequence_bwc, sequence_bwc)
        sequence_bwc = sequence_bwc + self.perceptron_layer_1(sequence_bwc)
        
        # Perform attention with the encoder output and the decoder sequence
        sequence_bwc = sequence_bwc + self.attention_layer_2(sequence_bwc, encoder_output_bwc)
        sequence_bwc = sequence_bwc + self.perceptron_layer_2(sequence_bwc)

        return sequence_bwc


class _BaseTransformerBlock(nn.Module):
    """ Base transformer block. """
    
    def __init__(self, params: TransformerBlockParameters, is_causal: bool = True):
        super(_BaseTransformerBlock, self).__init__()

        if params.attention == "metric":
            self.attention_layer = MetricSelfAttention(params, is_causal = is_causal)
        elif params.attention == "scaled_dot_product":
            self.attention_layer = ScaledDotProductAttention(params, is_causal = is_causal)
        else:
            raise ValueError(f"Invalid attention type: {params.attention}")
    

        self.perceptron_layer = make_perceptron_layer(params.coordinates)

    def forward(self, sequence_bwc: TensorFloat) -> TensorFloat:
        sequence_bwc = sequence_bwc + self.attention_layer(sequence_bwc, sequence_bwc)
        sequence_bwc = sequence_bwc + self.perceptron_layer(sequence_bwc)
        return sequence_bwc

class TransformerEncoderBlock(_BaseTransformerBlock):
    """ Transformer encoder block. """

    def __init__(self, params: TransformerBlockParameters):
        super(TransformerEncoderBlock, self).__init__(params, is_causal=False)

class TransformerDecoderBlock(_BaseTransformerBlock):
    """ Transformer decoder block. """

    def __init__(self, params: TransformerBlockParameters):
        super(TransformerDecoderBlock, self).__init__(params, is_causal=True)

