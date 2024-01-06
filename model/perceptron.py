import torch.nn as nn
from typing import Protocol
from core.types import TensorFloat


class PerceptronParameters(Protocol):
    coordinates: int
    bias: bool

class Perceptron(nn.Module):
    linear_expansion_dc: nn.Linear
    gelu_activation: nn.GELU
    linear_projections_cd: nn.Linear

    def __init__(self, params: PerceptronParameters):
        super().__init__()
        dimension = 4 * params.coordinates
        self.linear_expansion_dc = nn.Linear(params.coordinates, dimension, bias=params.bias)
        self.gelu_activation = nn.GELU()
        self.linear_projections_cd = nn.Linear(dimension, params.coordinates, bias=params.bias)


    def forward(self, in_sequence_bwc: TensorFloat) -> TensorFloat:
        sequence_bwd = self.linear_expansion_dc(in_sequence_bwc)
        sequence_bwd = self.gelu_activation(sequence_bwd)
        out_sequence_bwc = self.linear_projections_cd(sequence_bwd)
        return out_sequence_bwc
