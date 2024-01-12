
import torch.nn as nn
from typing import Protocol, Literal, Union
from core.types import TensorFloat
from model.self_attention import MetricSelfAttention, ScaledDotProductAttention
from core.logger import get_logger

logger = get_logger(__name__)


class TransformerBlockParameters(Protocol):
    coordinates: int
    words: int
    tokens: int
    attention: Literal["metric", "scaled_dot_product"]


class TransformerBlock(nn.Module):

    def __init__(self, params: TransformerBlockParameters):
        super(TransformerBlock, self).__init__()

        self.attention_layer = nn.Sequential(
            nn.LayerNorm(params.coordinates),
            MetricSelfAttention(params) if params.attention == "metric" else ScaledDotProductAttention(params),
        )

        self.perceptron_layer = nn.Sequential(
            nn.LayerNorm(params.coordinates),
            nn.Linear(params.coordinates, 4*params.coordinates),
            nn.GELU(),
            nn.Linear(4*params.coordinates, params.coordinates),
        )

    def forward(self, sequence_bwc: TensorFloat) -> TensorFloat:
        sequence_bwc = sequence_bwc + self.attention_layer(sequence_bwc)
        sequence_bwc = sequence_bwc + self.perceptron_layer(sequence_bwc)
        return sequence_bwc
