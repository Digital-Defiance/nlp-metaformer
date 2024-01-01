
import torch.nn as nn
from torch import Tensor
from typing import Protocol

from model.self_attention import SelfAttention
from model.perceptron import Perceptron as MLP

class TransformerBlockParameters(Protocol):
    coordinates: int
    words: int
    tokens: int

class TransformerBlock(nn.Module):

    norm1: nn.LayerNorm
    attention: SelfAttention

    norm2: nn.LayerNorm
    mlp: MLP


    def __init__(self, params: TransformerBlockParameters):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(params)
        self.mlp = MLP(params)
        self.norm1 = nn.LayerNorm(params.coordinates)
        self.norm2 = nn.LayerNorm(params.coordinates)

    def forward(self, in_sequence_bwc: Tensor):
        sequence_bwc = self.norm1(in_sequence_bwc)
        sequence_bwc = sequence_bwc + self.attention(sequence_bwc)
        sequence_bwc = self.norm2(sequence_bwc)
        out_sequence_bwc = sequence_bwc + self.mlp(sequence_bwc)
        return out_sequence_bwc