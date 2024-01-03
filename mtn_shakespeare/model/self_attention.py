
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
    attention_heads_dc: nn.Linear
    projection_cc: nn.Linear


    MASK_ww: Tensor
    NUMBER_OF_HEADS: int
    COORDINATES: int


    def __init__(self, params: SelfAttentionParameters):
        super(SelfAttention, self).__init__()

        self.COORDINATES = params.coordinates
        self.NUMBER_OF_HEADS = params.number_of_heads

        dimension = 2 * params.coordinates



        self.attention_heads_dc = nn.Linear(
            params.coordinates,
            dimension,
            bias=params.bias
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


        all_attention_vectors_bwd = self.attention_heads_dc(in_sequence_bwc)
        
        # split projections into three matrices
        #   - q_bwc contains all q vectors for the three heads
        #   - k_bwc contains all k vectors for the three heads
        #   - v_bwc contains all v vectors for the three heads
        q_bwc, k_bwc, v_bwc = all_attention_vectors_bwd.split(coordinates, dim=-1)
        
        
        # prepare for matrix multiplication
        k_dimension = coordinates // self.NUMBER_OF_HEADS
        q_b3wk = q_bwc.view(batch, words, self.NUMBER_OF_HEADS, k_dimension).transpose(1, 2)
        k_b3wk = k_bwc.view(batch, words, self.NUMBER_OF_HEADS, k_dimension).transpose(1, 2)
        v_b3wk = v_bwc.view(batch, words, self.NUMBER_OF_HEADS, k_dimension).transpose(1, 2)
      
        # perform matrix multiplication
        attention_scores_b3ww = q_b3wk @ k_b3wk.transpose(-1, -2)
        
        # prepare attention scores, scaling -> masking -> softmax
        attention_scores_b3ww = attention_scores_b3ww / math.sqrt(k_dimension)
        attention_scores_b3ww = attention_scores_b3ww.masked_fill(self.MASK_ww[:,:,:words,:words] == 0, float('-inf'))
        attention_scores_b3ww = F.softmax(attention_scores_b3ww, dim=-1)
        
        # produce the output sequence and shape it to the correct form
        out_sequence_b3wk = attention_scores_b3ww @ v_b3wk
        out_sequence_bw3k = out_sequence_b3wk.transpose(1, 2).contiguous()
        out_sequence_bwc = out_sequence_bw3k.view(batch, words, coordinates)
        
        # projection to mix the values 
        out_sequence_bwc = self.projection_cc(out_sequence_bwc)

        return out_sequence_bwc



    def _get_metric(self) -> Tensor:
        coordinates = self.COORDINATES
        WQ, WK, _ = self.attention_heads_dc.weight.transpose(-1, -2).split(coordinates , dim=-1)
        k_dimension = coordinates // self.NUMBER_OF_HEADS
        WQ = WQ.view(coordinates, self.NUMBER_OF_HEADS, k_dimension).transpose(0, 1)
        WK = WK.view(coordinates, self.NUMBER_OF_HEADS, k_dimension).transpose(0, 1)
        M = WQ @ WK.transpose(-1, -2)
        return M
    

