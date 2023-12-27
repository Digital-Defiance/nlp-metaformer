import torch
import torch.nn as nn
from torch import Tensor
from typing import Protocol

class SequenceEncoderParameters(Protocol):
    coordinates: int
    tokens: int
    words: int

class SequenceEncoder(nn.Module):
    vocabolary_enconding_tc: nn.Embedding
    positional_encoding_wc: nn.Embedding
    POSITION_INDICES_1w: Tensor
    
    def __init__(self, params: SequenceEncoderParameters) -> None:
        super(SequenceEncoder, self).__init__()
        self.vocabolary_enconding_tc = nn.Embedding(params.tokens, params.coordinates)
        self.positional_encoding_wc = nn.Embedding(params.words, params.coordinates)
        self.POSITION_INDICES_1w = torch.arange(0, params.words, dtype=torch.long).unsqueeze(0)

    def forward(self, sequence_bw: Tensor):
        sentence_tokens_bwc = self.vocabolary_enconding_tc(sequence_bw) # t = sequence_bw
        sentence_position_1wc = self.positional_encoding_wc(self.POSITION_INDICES_1w) # w = self.position_indices_1w
        out_sequence_bwc = sentence_tokens_bwc + sentence_position_1wc  
        return out_sequence_bwc