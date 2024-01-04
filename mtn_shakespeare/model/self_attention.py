
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




class MetricSelfAttention(nn.Module):
    projections_cd: nn.Linear
    projection_cc: nn.Linear

    MASK_ww: Tensor
    NUMBER_OF_HEADS: int
    COORDINATES: int


    def __init__(self, params: SelfAttentionParameters):
        super(MetricSelfAttention, self).__init__()

        self.COORDINATES = params.coordinates
        self.NUMBER_OF_HEADS = params.number_of_heads

        dimension = 2 * params.coordinates

        # TODO need to zero out some values here
        self.projections_cc = nn.Linear(
            self.COORDINATES,
            self.COORDINATES,
            bias = params.bias
        )

        self.pre_metric_tensors_nww = nn.Parameter(
            torch.tril(
                torch.ones(self.NUMBER_OF_HEADS, params.words, params.words)
            ),
        )

        self.projection_cc = nn.Linear(
            params.coordinates,
            params.coordinates,
            bias=params.bias
        )

        self.register_buffer(
            "MASK_11ww",
            torch
            .tril(torch.ones(1, 1, params.words, params.words))
            .view(1, 1, params.words, params.words)
        )


    def forward(self, in_sequence_bwc: Tensor) -> Tensor:
        batch, words, coordinates = in_sequence_bwc.size()
        k_dimension = coordinates // self.NUMBER_OF_HEADS
        pre_metric_tensors_nww = self.pre_metric_tensors_nww.masked_fill(self.MASK_ww[:,:,:words,:words] == 0, 0)
        metric_tensors_nww = pre_metric_tensors_nww @ pre_metric_tensors_nww.transpose(-1, -2)  # ensures symmetry and positive definiteness


        all_projections_bwc = self.projections_cc(in_sequence_bwc)
        all_projections_bnwk = all_projections_bwc.view(batch, words, self.NUMBER_OF_HEADS, k_dimension).transpose(1, 2)

        all_dot_products_bnww = all_projections_bnwk.transpose(-1, -2) @ metric_tensors_nww @ all_projections_bnwk
        all_dot_products_bnww = all_dot_products_bnww / math.sqrt(k_dimension)
        all_dot_products_bnww = all_dot_products_bnww.masked_fill(self.MASK_11ww[:,:,:words,:words] == 0, 0)

        nudged_vectors_bnwk = all_dot_products_bnww @ all_projections_bnwk
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
    

