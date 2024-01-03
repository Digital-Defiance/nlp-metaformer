
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Protocol


class SelfAttentionParameters(Protocol):
    bias: bool
    coordinates: int
    words: int
    number_of_heads: int






class SelfAttention(nn.Module):
    projections_cd: nn.Linear
    projection_cc: nn.Linear

    MASK_ww: Tensor
    NUMBER_OF_HEADS: int
    COORDINATES: int


    def __init__(self, params: SelfAttentionParameters):
        super(SelfAttention, self).__init__()

        self.COORDINATES = params.coordinates
        self.NUMBER_OF_HEADS = params.number_of_heads

        dimension = 2 * params.coordinates

        self.projections_cd = nn.Linear(
            self.COORDINATES,
            dimension,
            bias = params.bias
        )

        self.projection_cc = nn.Linear(
            params.coordinates,
            params.coordinates,
            bias=params.bias
        )

        self.register_buffer(
            "MASK_ww",
            torch
            .tril(torch.ones(params.words, params.words))
            .view(1, 1, params.words, params.words)
        )


    def forward(self, in_sequence_bwc: Tensor) -> Tensor:
        batch, words, coordinates = in_sequence_bwc.size()

        # vectors are projected twice into two same dimensional spaces
        all_projections_bwd = self.self.projections_cd(in_sequence_bwc)

        in_projections_bwc, out_projections_bwc =  all_projections_bwd.split(self.COORDINATES, dim=-1)

        k_dimension = coordinates // self.NUMBER_OF_HEADS
        in_projections_bnwk = in_projections_bwc.view(batch, words, self.NUMBER_OF_HEADS, k_dimension).transpose(1, 2)
        out_projections_bnwk = out_projections_bwc.view(batch, words, self.NUMBER_OF_HEADS, k_dimension).transpose(1, 2)

        all_dot_products_bnww = in_projections_bnwk @ in_projections_bnwk.transpose(-1, -2)
        all_dot_products_bnww = all_dot_products_bnww / math.sqrt(k_dimension)
        all_dot_products_bnww = all_dot_products_bnww.masked_fill(self.MASK_ww[:,:,:words,:words] == 0, 0)

        nudged_vectors_bnwk = all_dot_products_bnww @ out_projections_bnwk
        nudged_vectors_bwnk = nudged_vectors_bnwk.transpose(1, 2).contiguous()
        nudged_vectors_bwc = nudged_vectors_bwnk.view(batch, words, coordinates)

        out_sequence_bwc = self.projection_cc(nudged_vectors_bwc)

        return out_sequence_bwc



    def _get_metric(self) -> Tensor:
        raise NotImplementedError
        coordinates = self.COORDINATES
        WQ, WK, _ = self.attention_heads_dc.weight.transpose(-1, -2).split(coordinates , dim=-1)
        k_dimension = coordinates // self.NUMBER_OF_HEADS
        WQ = WQ.view(coordinates, self.NUMBER_OF_HEADS, k_dimension).transpose(0, 1)
        WK = WK.view(coordinates, self.NUMBER_OF_HEADS, k_dimension).transpose(0, 1)
        M = WQ @ WK.transpose(-1, -2)
        return M
    

