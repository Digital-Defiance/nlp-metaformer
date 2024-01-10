"""
This module contains the SequenceEncoder class.

"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Protocol


from core.constants import DEVICE
from core.types import TensorInt, TensorFloat



class SequenceEncoderParameters(Protocol):
    coordinates: int
    tokens: int
    words: int

class SequenceEncoder(nn.Module):

    vocabolary_enconding_tc: nn.Embedding
    positional_encoding_wc: nn.Embedding

    POSITION_INDICES_1w: TensorInt
    
    def __init__(self, params: SequenceEncoderParameters) -> None:
        super(SequenceEncoder, self).__init__()
        self.vocabolary_enconding_tc = nn.Embedding(params.tokens, params.coordinates)
        self.positional_encoding_wc = nn.Embedding(params.words, params.coordinates)
        self.register_buffer("POSITION_INDICES_1w", torch.arange(0, params.words, dtype=torch.long, device=DEVICE).unsqueeze(0))


    def forward(self, sequence_bw: TensorInt) -> TensorFloat:

        _, words = sequence_bw.size()
        max_words = self.POSITION_INDICES_1w.size(1)


        assert words <= max_words, f"words={words} > self.POSITION_INDICES_1w.size(1)={self.POSITION_INDICES_1w.size(1)}"


        sentence_tokens_bwc = self.vocabolary_enconding_tc(sequence_bw)
        sentence_position_1wc = self.positional_encoding_wc(self.POSITION_INDICES_1w[:, :words])
        return sentence_tokens_bwc + sentence_position_1wc