


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


class EmbeddingsConfig(BaseModel):
    dim: int = 128
    vocab_size: int = 50257
    ctx_win: int = 64
    
    def to_model(self):
        return Embedder(self)

class Embedder(nn.Module):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__()
        self.vocabolary = nn.Embedding(config.vocab_size, config.dim)
        self.pos_encoding = nn.Embedding(config.ctx_win, config.dim)

    def forward(self, x_bc: Tensor):
        C = x_bc.size(1)
        ctx = torch.arange(0, C).to("cuda")
        return self.vocabolary(x_bc) + self.pos_encoding(ctx)