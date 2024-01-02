
import torch.nn as nn
from torch import Tensor
from typing import Protocol

from model.self_attention import SelfAttention
from model.perceptron import Perceptron as MLP

TensorInt = Tensor
TensorFloat = Tensor

class TransformerBlockParameters(Protocol):
    coordinates: int
    words: int
    tokens: int

class TransformerBlock(nn.Module):

    layer_norm1_c: nn.LayerNorm
    self_attention: SelfAttention

    layer_norm2_c: nn.LayerNorm
    perceptron_layer: MLP


    def __init__(self, params: TransformerBlockParameters):
        super(TransformerBlock, self).__init__()

        self.layer_norm1_c = nn.LayerNorm(params.coordinates)
        self.self_attention = SelfAttention(params)

        self.layer_norm2_c = nn.LayerNorm(params.coordinates)
        self.perceptron_layer = MLP(params)

    def forward(self, in_sequence_bwc: TensorFloat) -> TensorFloat:
        sequence_bwc = self.layer_norm1_c(in_sequence_bwc)
        sequence_bwc = sequence_bwc + self.self_attention(sequence_bwc)
        sequence_bwc = self.layer_norm2_c(sequence_bwc)
        out_sequence_bwc = sequence_bwc + self.perceptron_layer(sequence_bwc)
        return out_sequence_bwc
