

import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from pydantic import BaseModel
from pydantic import BaseModel, Field, Extra, SkipValidation
from typing import Annotated
from torch import nn
from torch import Tensor
from typing import Literal
from torch.nn import functional as F
from pipelines.commons import make_linear

from pipelines.backbone.embedder import EmbeddingsConfig
from pipelines.backbone.attention import AttentionConfig
from pipelines.backbone.feedforward import FeedForwardConfig


class TransformerConfig(BaseModel):
    embeddings_cfg: EmbeddingsConfig = EmbeddingsConfig()
    body: list[AttentionConfig | FeedForwardConfig]
    
    def to_model(self):
        return Transformer(self)

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.embedder = cfg.embeddings_cfg.to_model()
        self.body = nn.Sequential()
        for i, layer in enumerate(cfg.body):
            module = layer.to_model(cfg.embeddings_cfg)
            self.body.add_module(f"layer_{i}", module)

    def forward(self, x_bc: Tensor) -> Tensor:
        x_bcd = self.embedder(x_bc)
        x_bcd = self.body(x_bcd)
        return x_bcd
