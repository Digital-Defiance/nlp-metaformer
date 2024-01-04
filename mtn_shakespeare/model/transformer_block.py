
import torch.nn as nn
from torch import Tensor
from typing import Protocol

from model.self_attention import MetricSelfAttention
from model.perceptron import Perceptron as MLP
from model.l2_norm import L2Normalization


TensorInt = Tensor
TensorFloat = Tensor

class TransformerBlockParameters(Protocol):
    coordinates: int
    words: int
    tokens: int

class TransformerBlock(nn.Module):

    layer_norm1_c: L2Normalization
    self_attention: MetricSelfAttention

    layer_norm2_c: L2Normalization
    perceptron_layer: MLP


    def __init__(self, params: TransformerBlockParameters):
        super(TransformerBlock, self).__init__()

        self.layer_norm1_c = L2Normalization(params)
        self.self_attention = MetricSelfAttention(params)

        self.layer_norm2_c = L2Normalization(params)
        self.perceptron_layer = MLP(params)

    def forward(self, in_sequence_bwc: TensorFloat) -> TensorFloat:
        sequence_bwc = self.layer_norm1_c(in_sequence_bwc)
        sequence_bwc = sequence_bwc + self.self_attention(sequence_bwc)
        sequence_bwc = self.layer_norm2_c(sequence_bwc)
        out_sequence_bwc = sequence_bwc + self.perceptron_layer(sequence_bwc)
        return out_sequence_bwc
