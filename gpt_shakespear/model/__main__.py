
import torch
from torch import Tensor
import torch.nn as nn
from model.sequence_encoder import SequenceEncoder
from model.transformer_block import TransformerBlock
from typing import Protocol


TensorInt = Tensor
TensorFloat = Tensor


class ModelParameters(Protocol):
    """
    Represents the parameters of a model.

    Attributes:
        coordinates (int): The dimension of a vector embedding.
        tokens (int): The number of tokens in the vocabulary.
        words (int): The maximum number of words in a sentence (context window).
        number_of_blocks (int): The number of blocks in the model.
    """
    coordinates: int
    tokens: int
    words: int
    number_of_blocks: int

class NanoGPT(nn.Module):
    """
    NanoGPT model for sequence generation.
    
    Args:
        params (ModelParameters): The parameters for the model.
    """
    sequence_encoder: SequenceEncoder
    transformer_blocks: nn.Sequential
    layer_norm_c: nn.LayerNorm
    language_model_weights_tc: nn.Linear

    def __init__(self, params: ModelParameters):
        super(NanoGPT, self).__init__()

        self.sequence_encoder = SequenceEncoder(params)

        transformer_blocks = [TransformerBlock(params) for _ in range(params.number_of_blocks)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.layer_norm_c = nn.LayerNorm(params.coordinates)
        self.language_model_weights_tc = nn.Linear(params.coordinates, params.tokens, bias=False)

    def forward(self, in_sequence_bw: TensorInt) -> TensorFloat:
        """
        Forward pass of the model.
        
        Args:
            in_sequence_bw (torch.Tensor): The input sequence tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: The output logits tensor of shape (batch_size, sequence_length, num_tokens).
        """
        sequence_bwc = self.sequence_encoder(in_sequence_bw)
        sequence_bwc = self.transformer_blocks(sequence_bwc)
        sequence_bwc = self.layer_norm_c(sequence_bwc)
        logits_bwt = self.language_model_weights_tc(sequence_bwc)
        return logits_bwt


