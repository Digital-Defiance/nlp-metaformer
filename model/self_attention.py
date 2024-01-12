
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Protocol
from abc import ABC, abstractmethod
from core.constants import DEVICE



# --- contracts --- #

class SelfAttentionParameters(Protocol):
    bias: bool
    coordinates: int
    words: int
    number_of_heads: int


def create_parameter(*shape: tuple[int, ...], requires_grad: bool = True) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape), requires_grad=requires_grad)


class SelfAttention(ABC):
    MASK_ww: Tensor
    NUMBER_OF_HEADS: int
    COORDINATES: int
    K_DIMENSION: int


    def create_parameter(self, shape: tuple[int, ...], requires_grad: bool = True) -> nn.Parameter:
        return nn.Parameter(torch.randn(*shape), requires_grad=requires_grad)


    def set_common_parameters(self: nn.Module, params: SelfAttentionParameters):
        self.COORDINATES = params.coordinates
        self.NUMBER_OF_HEADS = params.number_of_heads
        self.K_DIMENSION = self.COORDINATES // self.NUMBER_OF_HEADS
        self.SQRT_K_DIMENSION = math.sqrt(self.K_DIMENSION)

        self.mixer_cc = nn.Linear(params.coordinates, params.coordinates, bias=params.bias)

        self.register_buffer(
            "MASK_11ww",
            torch
              .tril(torch.ones(params.words, params.words))
              .view(1, 1, params.words, params.words)
        )

    @abstractmethod
    def __init__(self, params: SelfAttentionParameters):
        ...

    @abstractmethod
    def forward(self, in_sequence_bwc: Tensor) -> Tensor:
        ...

    @abstractmethod
    def get_metric(self) -> Tensor:
        ...




# --- implementations --- #


class MetricSelfAttention(nn.Module, SelfAttention):
    MASK_ww: Tensor
    NUMBER_OF_HEADS: int
    COORDINATES: int
    K_DIMENSION: int

    def __init__(self, params: SelfAttentionParameters):
        super(MetricSelfAttention, self).__init__()

        self.set_common_parameters(params)

        buffers = {
            "INDICES": torch.triu_indices(row=self.K_DIMENSION, col=self.K_DIMENSION, offset=1, device=DEVICE),
        }

        learnable_parameters = {
            "projection_1nck": torch.randn(1, self.NUMBER_OF_HEADS, self.COORDINATES, self.K_DIMENSION),
            "halves": torch.randn(self.NUMBER_OF_HEADS, self.K_DIMENSION*(self.K_DIMENSION + 1) // 2 - self.K_DIMENSION),
            "diagonals_nk": torch.randn(self.NUMBER_OF_HEADS, self.K_DIMENSION),
        }

        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)

        for name, tensor in learnable_parameters.items():
            self.register_parameter(name, nn.Parameter(tensor))

   
    def get_metric(self) -> Tensor:
        metric_tensors_nkk = torch.zeros(self.NUMBER_OF_HEADS, self.K_DIMENSION, self.K_DIMENSION)
        metric_tensors_nkk[:, self.INDICES[0], self.INDICES[1]] = self.halves
        metric_tensors_nkk = metric_tensors_nkk + metric_tensors_nkk.transpose(-1, -2)
        metric_tensors_nkk = torch.diagonal_scatter(
            metric_tensors_nkk, self.diagonals_nk,
            offset=0, dim1=1, dim2=2
        )
        return metric_tensors_nkk

    def forward(self, in_sequence_bwc: Tensor) -> Tensor:
        batch, words, coordinates = in_sequence_bwc.size()
        metric_tensors_nkk = self.get_metric()
        metric_tensors_1nkk = metric_tensors_nkk.unsqueeze(0)

        in_sequence_b1wc = in_sequence_bwc.unsqueeze(1)
        all_projections_bnwk = in_sequence_b1wc @ self.projection_1nck

        all_dot_products_bnww = all_projections_bnwk @ metric_tensors_1nkk @ all_projections_bnwk.transpose(-1, -2)
        all_scaled_dot_products_bnww = all_dot_products_bnww / self.SQRT_K_DIMENSION
        all_scaled_dot_products_bnww = all_scaled_dot_products_bnww.masked_fill(self.MASK_11ww[:,:,:words,:words] == 0, float('-inf'))
        all_scaled_dot_products_bnww = F.softmax(all_scaled_dot_products_bnww, dim=-1)

        nudged_vectors_bnwk = all_dot_products_bnww @ all_projections_bnwk
        nudged_vectors_bwnk = nudged_vectors_bnwk.transpose(1, 2).contiguous()
        nudged_vectors_bwc = nudged_vectors_bwnk.view(batch, words, coordinates)

        out_sequence_bwc = self.mixer_cc(nudged_vectors_bwc)
        return out_sequence_bwc




class ScaledDotProductAttention(nn.Module, SelfAttention):
    MASK_ww: Tensor
    NUMBER_OF_HEADS: int
    COORDINATES: int
    K_DIMENSION: int
    D_DIMENSION: int

    projection_cc: nn.Parameter
    attention_heads_cd: nn.Parameter

    def __init__(self, params: SelfAttentionParameters):
        super(ScaledDotProductAttention, self).__init__()
        self.set_common_parameters(params)
        self.D_DIMENSION = 3 * params.coordinates
        learnable_parameters = {
            "attention_heads_cd": torch.randn(params.coordinates, self.D_DIMENSION),
            "projection_cc": torch.randn(params.coordinates, params.coordinates),
        }
    
        for name, tensor in learnable_parameters.items():
            self.register_parameter(name, nn.Parameter(tensor))

    def forward(self, in_sequence_bwc: Tensor) -> Tensor:
 
        batch, words, _ = in_sequence_bwc.size()

        all_attention_vectors_bwd = in_sequence_bwc @ self.attention_heads_cd
        
        # split projections into three matrices
        #   - q_bwc contains all q vectors for the three heads
        #   - k_bwc contains all k vectors for the three heads
        #   - v_bwc contains all v vectors for the three heads
        q_bwc, k_bwc, v_bwc = all_attention_vectors_bwd.split(self.COORDINATES, dim=-1)
        
        
        # prepare for matrix multiplication by splitting the heads
        q_bnwk = q_bwc.view(batch, words, self.NUMBER_OF_HEADS, self.K_DIMENSION).transpose(1, 2)
        k_bnwk = k_bwc.view(batch, words, self.NUMBER_OF_HEADS, self.K_DIMENSION).transpose(1, 2)
        v_bnwk = v_bwc.view(batch, words, self.NUMBER_OF_HEADS, self.K_DIMENSION).transpose(1, 2)
      
        # perform matrix multiplication (dot product)
        attention_scores_bnww = q_bnwk @ k_bnwk.transpose(-1, -2)
        
        # prepare attention scores, scaling -> masking -> softmax
        scaled_attention_scores_bnww = attention_scores_bnww / self.SQRT_K_DIMENSION
        scaled_attention_scores_bnww = scaled_attention_scores_bnww.masked_fill(self.MASK_11ww[:,:,:words,:words] == 0, float('-inf'))
        scaled_attention_scores_bnww = F.softmax(scaled_attention_scores_bnww, dim=-1)

        # produce the output sequence and shape it to the correct form
        out_sequence_bnwk = scaled_attention_scores_bnww @ v_bnwk
        out_sequence_bwnk = out_sequence_bnwk.transpose(1, 2).contiguous()
        out_sequence_bwc = out_sequence_bwnk.view(batch, words, self.COORDINATES)
        
        # projection to mix the values 
        out_sequence_bwc = self.mixer_cc(out_sequence_bwc)
        return out_sequence_bwc


    def get_metric(self) -> Tensor:
        raise NotImplementedError
        coordinates = self.COORDINATES
        WQ, WK, _ = self.attention_heads_dc.weight.transpose(-1, -2).split(coordinates , dim=-1)
        k_dimension = coordinates // self.NUMBER_OF_HEADS
        WQ = WQ.view(coordinates, self.NUMBER_OF_HEADS, k_dimension).transpose(0, 1)
        WK = WK.view(coordinates, self.NUMBER_OF_HEADS, k_dimension).transpose(0, 1)
        M = WQ @ WK.transpose(-1, -2)
        return M

