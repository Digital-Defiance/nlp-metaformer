
import torch
import torch.nn as nn
from model.sequence_encoder import SequenceEncoder
from model.transformer_block import TransformerBlock
from typing import Protocol


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
    transformers: nn.Sequential
    norm: nn.LayerNorm
    lm_weights: nn.Linear

    def __init__(self, params: ModelParameters):
        super(NanoGPT, self).__init__()
        self.sequence_encoder = SequenceEncoder(params)
        transformers_generator = (TransformerBlock(params) for _ in range(params.number_of_blocks))
        self.transformers = nn.Sequential(*transformers_generator)
        self.norm = nn.LayerNorm(params.coordinates)
        self.lm_weights = nn.Linear(params.coordinates, params.tokens, bias=False)

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the model.
        
        Returns:
            int: The total number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.sequence_encoder.positional_encoding_wc.weight.numel()
        return n_params

    def forward(self, in_sequence_bw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            in_sequence_bw (torch.Tensor): The input sequence tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: The output logits tensor of shape (batch_size, sequence_length, num_tokens).
        """
        sequence_bwc = self.sequence_encoder(in_sequence_bw)
        sequence_bwc = self.transformers(sequence_bwc)
        sequence_bwc = self.norm(sequence_bwc)
        logits_bw = self.lm_weights(sequence_bwc)
        return logits_bw


