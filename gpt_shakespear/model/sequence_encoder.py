"""
This module contains the SequenceEncoder class.

"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Protocol


TensorInt = Tensor
TensorFloat = Tensor

class SequenceEncoderParameters(Protocol):
    """
    Represents the parameters for a sequence encoder.

    Attributes:
        coordinates (int): The number of coordinates in the word embeddings.
        tokens (int): The number of tokens in the vocabolary.
        words (int): The number of words in the sequence.
    """

    coordinates: int
    tokens: int
    words: int

class SequenceEncoder(nn.Module):
    """
    A module that encodes a sequence using token and positional encodings.

    Attributes:
        vocabolary_enconding_tc (nn.Embedding): The embedding layer for token encoding.
        positional_encoding_wc (nn.Embedding): The embedding layer for positional encoding.
        POSITION_INDICES_1w (Tensor): The tensor containing positional indices.

    Args:
        params (SequenceEncoderParameters): The parameters for the SequenceEncoder.

    Returns:
        None
    """

    vocabolary_enconding_tc: nn.Embedding
    positional_encoding_wc: nn.Embedding
    POSITION_INDICES_1w: TensorInt
    
    def __init__(self, params: SequenceEncoderParameters) -> None:
        """
        Initializes the SequenceEncoder module.

        Args:
            params (SequenceEncoderParameters): The parameters for the SequenceEncoder.

        Returns:
            None
        """
        super(SequenceEncoder, self).__init__()
        self.vocabolary_enconding_tc = nn.Embedding(params.tokens, params.coordinates)
        self.positional_encoding_wc = nn.Embedding(params.words, params.coordinates)
        self.POSITION_INDICES_1w = torch.arange(0, params.words, dtype=torch.long).unsqueeze(0)

    def forward(self, sequence_bw: TensorInt) -> TensorFloat:
        """
        Performs the forward pass of the SequenceEncoder.

        Args:
            sequence_bw (Tensor): The input sequence tensor.

        Returns:
            Tensor: The output sequence tensor.
        """
        sentence_tokens_bwc = self.vocabolary_enconding_tc(sequence_bw) # t = sequence_bw
        sentence_position_1wc = self.positional_encoding_wc(self.POSITION_INDICES_1w) # w = self.position_indices_1w
        out_sequence_bwc = sentence_tokens_bwc + sentence_position_1wc  
        return out_sequence_bwc