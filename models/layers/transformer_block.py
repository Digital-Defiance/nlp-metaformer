
import torch.nn as nn
from torch import Tensor
from typing import Protocol, Literal, Union

from models.layers.self_attention import MetricSelfAttention, ScaledDotProductAttention
from models.layers.perceptron import Perceptron as MLP


TensorInt = Tensor
TensorFloat = Tensor


class TransformerBlockParameters(Protocol):
    coordinates: int
    words: int
    tokens: int

    attention: Literal["metric", "scaled_dot_product"]


class TransformerBlock(nn.Module):

    layer_norm1_c: nn.LayerNorm
    self_attention: Union[MetricSelfAttention, ScaledDotProductAttention]

    layer_norm2_c: nn.LayerNorm
    perceptron_layer: nn.LayerNorm


    def __init__(self, params: TransformerBlockParameters):
        super(TransformerBlock, self).__init__()


        self.layer_norm1_c = nn.LayerNorm(params)

        if params.attention == "metric":
            self.self_attention = MetricSelfAttention(params)
        elif params.attention == "scaled_dot_product":
            self.self_attention = ScaledDotProductAttention(params)

        self.layer_norm2_c = nn.LayerNorm(params)
        self.perceptron_layer = MLP(params)

    def forward(self, in_sequence_bwc: TensorFloat) -> TensorFloat:
        sequence_bwc = self.layer_norm1_c(in_sequence_bwc)
        sequence_bwc = sequence_bwc + self.self_attention(sequence_bwc)
        sequence_bwc = self.layer_norm2_c(sequence_bwc)
        out_sequence_bwc = sequence_bwc + self.perceptron_layer(sequence_bwc)
        return out_sequence_bwc
