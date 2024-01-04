
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Protocol
import torch

class L2NormalizationParameters(Protocol):
    coordinates: int

class L2Normalization(nn.Module):
    def __init__(self, params: L2NormalizationParameters):
        super(L2Normalization, self).__init__()

        self.linear_scale_11c = nn.Parameter(torch.ones(1, 1, params.coordinates))   
        self.bias_11c = nn.Parameter(torch.zeros(1, 1, params.coordinates))
        self.layernorm = nn.LayerNorm(params.coordinates)

    def forward(self, sequence_bwc: Tensor) -> Tensor:
        # normalized_sequence_bwc = F.normalize(sequence_bwc, p=2, dim=2)
        # normalized_sequence_bwc =  normalized_sequence_bwc * self.linear_scale_11c + self.bias_11c
        return self.layernorm(sequence_bwc)
    


