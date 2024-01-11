"""
This module contains the SequenceEncoder class.

"""
import torch
import torch.nn as nn
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

        buffers = {
            "POSITION_INDICES_1w": torch.arange(0, params.words, dtype=torch.long, device=DEVICE).unsqueeze(0)
        }

        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)

    def forward(self, sequence_bw: TensorInt) -> TensorFloat:
        _, words = sequence_bw.size()
        max_words = self.POSITION_INDICES_1w.size(1)
        assert words <= max_words, f"Size of sequence exceeds context window size of {max_words}"
        sentence_tokens_bwc = self.vocabolary_enconding_tc(sequence_bw)
        sentence_position_1wc = self.positional_encoding_wc(self.POSITION_INDICES_1w[:, :words])
        return sentence_tokens_bwc + sentence_position_1wc
