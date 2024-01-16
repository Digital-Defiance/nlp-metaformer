
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


    def set_common_parameters(self: nn.Module, params: SelfAttentionParameters, is_causal: bool):
        self.COORDINATES = params.coordinates
        self.NUMBER_OF_HEADS = params.number_of_heads
        self.K_DIMENSION = self.COORDINATES // self.NUMBER_OF_HEADS
        self.SQRT_K_DIMENSION = math.sqrt(self.K_DIMENSION)
        self.IS_CAUSAL = is_causal

        self.norm = nn.LayerNorm(params.coordinates)

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
    def forward(self, in_sequence1_bwc: Tensor, in_sequence2_bwc: Tensor) -> Tensor:
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

    def __init__(self, params: SelfAttentionParameters, is_causal: bool):
        super(MetricSelfAttention, self).__init__()

        self.set_common_parameters(params, is_causal)

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
        metric_tensors_nkk = torch.zeros(self.NUMBER_OF_HEADS, self.K_DIMENSION, self.K_DIMENSION, device=DEVICE)
        metric_tensors_nkk[:, self.INDICES[0], self.INDICES[1]] = self.halves
        metric_tensors_nkk = metric_tensors_nkk + metric_tensors_nkk.transpose(-1, -2)
        metric_tensors_nkk = torch.diagonal_scatter(
            metric_tensors_nkk, self.diagonals_nk,
            offset=0, dim1=1, dim2=2
        ).to(DEVICE)
        return metric_tensors_nkk

    def forward(self, in_sequence1_bwc: Tensor, in_sequence2_bwc: Tensor) -> Tensor:
        batch1, words1, coordinates1 = in_sequence1_bwc.size()
        batch2, words2, coordinates2 = in_sequence2_bwc.size()

        assert batch1 == batch2, "Batch size of both sequences must be the same"
        assert coordinates1 == coordinates2, "Coordinates of both sequences must be the same"
        assert words1 == words2, "Words of both sequences must be the same"

        in_sequence1_bwc = self.norm(in_sequence1_bwc)
        in_sequence2_bwc = self.norm(in_sequence2_bwc)

        metric_tensors_nkk = self.get_metric()
        metric_tensors_1nkk = metric_tensors_nkk.unsqueeze(0)

        in_sequence1_b1wc = in_sequence1_bwc.unsqueeze(1)
        in_sequence2_b1wc = in_sequence2_bwc.unsqueeze(1)
    
        all_projections1_bnwk = in_sequence1_b1wc @ self.projection_1nck
        all_projections2_bnwk = in_sequence2_b1wc @ self.projection_1nck

        all_dot_products_bnww = all_projections1_bnwk @ metric_tensors_1nkk @ all_projections2_bnwk.transpose(-1, -2)
        all_scaled_dot_products_bnww = all_dot_products_bnww / self.SQRT_K_DIMENSION

        if self.IS_CAUSAL:
            all_scaled_dot_products_bnww = all_scaled_dot_products_bnww.masked_fill(
                self.MASK_11ww[:,:,:words1,:words1] == 0,
                float('-inf')
            )
        
        all_scaled_dot_products_bnww = F.softmax(all_scaled_dot_products_bnww, dim=-1)

        nudged_vectors_bnwk = all_dot_products_bnww @ all_projections1_bnwk
        nudged_vectors_bwnk = nudged_vectors_bnwk.transpose(1, 2).contiguous()
        nudged_vectors_bwc = nudged_vectors_bwnk.view(batch1, words1, coordinates1)

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

    def __init__(self, params: SelfAttentionParameters, is_decoder: bool):
        super(ScaledDotProductAttention, self).__init__()
        self.set_common_parameters(params, is_decoder)
        self.D_DIMENSION = 3 * params.coordinates
        learnable_parameters = {
            "attention_heads_cd": torch.randn(params.coordinates, self.D_DIMENSION),
            "q_projection_1nck": torch.randn(1, params.number_of_heads, params.coordinates, self.K_DIMENSION),
            "k_projection_1nck": torch.randn(1, params.number_of_heads, params.coordinates, self.K_DIMENSION),
            "v_projection_1nck": torch.randn(1, params.number_of_heads, params.coordinates, self.K_DIMENSION),
            "projection_cc": torch.randn(params.coordinates, params.coordinates),
        }
    
        for name, tensor in learnable_parameters.items():
            self.register_parameter(name, nn.Parameter(tensor))

    def forward(self, in_sequence1_bwc: Tensor, in_sequence2_bwc: Tensor) -> Tensor:


 
        batch, words, _ = in_sequence1_bwc.size()
        assert in_sequence1_bwc.size() == in_sequence2_bwc.size(), "Both sequences must have the same shape"


        in_sequence1_bwc = self.norm(in_sequence1_bwc)
        in_sequence2_bwc = self.norm(in_sequence2_bwc)

        sequence1_b1wc = in_sequence1_bwc.unsqueeze(1)
        sequence2_b1wc = in_sequence2_bwc.unsqueeze(1)

        q_bnwk = sequence1_b1wc @ self.q_projection_1nck
        k_bnwk = sequence2_b1wc @ self.k_projection_1nck
        v_bnwk = sequence2_b1wc @ self.v_projection_1nck
      
        # perform matrix multiplication (dot product)
        attention_scores_bnww = q_bnwk @ k_bnwk.transpose(-1, -2)
        
        # prepare attention scores, scaling -> masking -> softmax
        scaled_attention_scores_bnww = attention_scores_bnww / self.SQRT_K_DIMENSION
        if self.IS_DECODER:
            scaled_attention_scores_bnww = scaled_attention_scores_bnww.masked_fill(
                self.MASK_11ww[:,:,:words,:words] == 0,
                float('-inf')
            )

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

