
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Protocol


class SelfAttentionParameters(Protocol):
    coordinates: int
    words: int

class SelfAttention(nn.Module):
    COORDINATES: int
    attention_heads_dc: nn.Linear
    projection_1cc: nn.Linear
    BIAS_ww: Tensor
    metrics_31ww: Tensor

    def __init__(self, params: SelfAttentionParameters):
        """
        Initializes the SelfAttention module.

        Args:
            params (SelfAttentionParameters): The parameters for the SelfAttention module.
        """
        super(SelfAttention, self).__init__()
        self.COORDINATES = params.coordinates

        # d = 3*c
        self.attention_heads_dc = nn.Linear(params.coordinates, 3*params.coordinates, bias=False)
        self.projection_1cc = nn.Linear(params.coordinates, params.coordinates, bias=False)
        self.register_buffer(
            "BIAS_ww",
            torch
            .tril(torch.ones(params.words, params.words))
            .view(1, 1, params.words, params.words)
        )


    def forward(self, in_sequence_bwc: Tensor) -> Tensor:
        """
        Performs forward pass through the SelfAttention module.

        Args:
            in_sequence_bwc (Tensor): The input sequence tensor of shape (batch, words, coordinates).

        Returns:
            Tensor: The output sequence tensor of shape (batch, words, coordinates).
        """
        batch, words, coordinates = in_sequence_bwc.size()

        # d = 3*c where c = coordinates
        # perform projections
        all_vectors_bwd = self.attention_heads_dc.forward(in_sequence_bwc)
        
        # split projections into three matrices
        #   - q_bwc contains all q vectors for the three heads
        #   - k_bwc contains all k vectors for the three heads
        #   - v_bwc contains all v vectors for the three heads
        q_bwc, k_bwc, v_bwc = all_vectors_bwd.split(coordinates, dim=-1)
        
        
        # prepare for matrix multiplication
        q_b3wk = q_bwc.view(batch, words, 3, coordinates // 3).transpose(1, 2)
        k_b3wk = k_bwc.view(batch, words, 3, coordinates // 3).transpose(1, 2)
        v_b3wk = v_bwc.view(batch, words, 3, coordinates // 3).transpose(1, 2)
      
        # perform matrix multiplication
        attention_scores_b3ww = q_b3wk @ k_b3wk.transpose(-1, -2)
        
        # prepare attention scores, scaling -> masking -> softmax
        attention_scores_b3ww = attention_scores_b3ww / math.sqrt(v_b3wk.size(-1))
        attention_scores_b3ww = attention_scores_b3ww.masked_fill(self.BIAS_ww[:,:,:words,:words] == 0, float('-inf'))
        attention_scores_b3ww = F.softmax(attention_scores_b3ww, dim=-1)
        
        # produce the output sequence and shape it to the correct form
        out_sequence_b3wk = attention_scores_b3ww @ v_b3wk
        out_sequence_bw3k = out_sequence_b3wk.transpose(1, 2).contiguous()
        out_sequence_bwc = out_sequence_bw3k.view(batch, words, coordinates)
        
        # projection to mix the values 
        out_sequence_bwc = self.projection_1cc(out_sequence_bwc)

        return out_sequence_bwc



    def _get_metric(self):
        """
        Computes the metric for the SelfAttention module.

        Returns:
            Tensor: The computed metric tensor.
        """
        coordinates = self.COORDINATES
        Wq, Wk, _ = self.attention_heads_dc.weight.transpose(-1, -2).split(coordinates , dim=-1)
        Wq = Wq.view(coordinates, 3, coordinates // 3).transpose(0, 1)
        Wk = Wk.view(coordinates, 3, coordinates // 3).transpose(0, 1)

        M = Wq @ Wk.transpose(-1, -2) # 3 x (wordsxwords)
        M = M.unsqueeze(1) # 3 x 1 x (wordsxwords)
        return M
    




