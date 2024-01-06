import torch.nn as nn
from torch import Tensor
from typing import Protocol


TensorFloat = Tensor

class PerceptronParameters(Protocol):
    coordinates: int

class Perceptron(nn.Module):
    linear_expansion_dc: nn.Linear
    gelu_activation: nn.GELU
    linear_projections_cd: nn.Linear

    def __init__(self, params: PerceptronParameters):
        super().__init__()
        self.linear_expansion_dc = nn.Linear(params.coordinates, 4 * params.coordinates, bias=False)
        self.gelu_activation = nn.GELU()
        self.linear_projections_cd = nn.Linear(4 * params.coordinates, params.coordinates, bias=False)


    def forward(self, in_sequence_bwc: TensorFloat) -> TensorFloat:
        sequence_bwd = self.linear_expansion_dc(in_sequence_bwc)
        sequence_bwd = self.gelu_activation(sequence_bwd)
        out_sequence_bwc = self.linear_projections_cd(sequence_bwd)
        return out_sequence_bwc
