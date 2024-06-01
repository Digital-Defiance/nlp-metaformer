

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from pydantic import BaseModel
from typing import Annotated
from torch import nn
from torch import Tensor
from typing import Literal
from pipelines.commons import make_linear
from pipelines.backbone.embedder import EmbeddingsConfig
from pipelines.backbone.residual import Residual


class AttentionConfig(BaseModel):
    number_of_heads: int = 6
    kind: Literal["quadratic", "metric", "scaled_dot_product"] = "metric"
    norm_input: Literal["LayerNorm"] | None = "LayerNorm"
    residual: bool = True
    masked: bool = False

    def to_model(self, embedd_config: EmbeddingsConfig):
        return Attention(self, embedd_config)



class Attention(nn.Module):
    
    def __init__(self, cfg: AttentionConfig, embedd_cfg: EmbeddingsConfig):
        super().__init__()
        
        if cfg.kind == "quadratic":
            self.attention = QuadraticAttention(cfg, embedd_cfg)
        elif cfg.kind == "metric":
            self.attention = MetricAttention(cfg, embedd_cfg)
        else:
            raise NotImplementedError

        if cfg.norm_input is not None:
            self.attention = nn.Sequential(
                nn.LayerNorm(embedd_cfg.dim),
                self.attention
            )
            
        if cfg.residual:
            self.attention = Residual(self.attention)
            
    def forward(self, x_bcd):
        return self.attention(x_bcd)
        

class QuadraticAttention(nn.Module):
    def __init__(self, config: AttentionConfig, embedd_config: EmbeddingsConfig):
        super().__init__()
        k = embedd_config.dim // config.number_of_heads
        self.coef_1nkk = make_linear(1, config.number_of_heads, k, k)
        self.proj_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k)
        self.mixer_dd = make_linear(embedd_config.dim, embedd_config.dim)

    def forward(self, x_bcd: Tensor):
        x_b1cd = x_bcd.unsqueeze(1)
        x_bnck = x_b1cd @ self.proj_1ndk
        scores_bncc = x_bnck @ self.coef_1nkk @ x_bnck.transpose(-1, -2)
        scores_bncc = F.softmax(scores_bncc, dim=-1)
        x_bnck = scores_bncc @ x_bnck
        x_bcd = x_bnck.reshape(x_bcd.shape)
        x_bcd = x_bcd @ self.mixer_dd
        return x_bcd


class MetricAttention(nn.Module):
    def __init__(self, config: AttentionConfig, embedd_config: EmbeddingsConfig):
        super().__init__()
        k = embedd_config.dim // config.number_of_heads
        self.coef_1nkk = make_linear(1, config.number_of_heads, k, k)
        self.proj_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k)
        self.mixer_dd = make_linear(embedd_config.dim, embedd_config.dim)

    def forward(self, x_bcd: Tensor):
        return x_bcd
        x_b1cd = x_bcd.unsqueeze(1)
        x_bnck = x_b1cd @ self.proj_1ndk
        metrics_1nkk = (self.coef_1nkk + self.coef_1nkk.transpose(-1, -2)) / 2
        scores_bncc = x_bnck @ metrics_1nkk @ x_bnck.transpose(-1, -2)
        scores_bncc = F.softmax(scores_bncc, dim=-1)
        x_bnck = scores_bncc @ x_bnck
        x_bcd = x_bnck.reshape(x_bcd.shape)
        x_bcd = x_bcd @ self.mixer_dd
        return x_bcd

class ScaledDotProductAttention(nn.Module):
    ...


class Identity(nn.Module):
    def __init__(self, config: AttentionConfig, embedd_config: EmbeddingsConfig):
        super().__init__()
        k = embedd_config.dim // config.number_of_heads
        self.coef_1nkk = make_linear(1, config.number_of_heads, k, k)
        self.proj_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k)
        self.mixer_dd = make_linear(embedd_config.dim, embedd_config.dim)

    def forward(self, x_bcd: Tensor):
        x_b1cd = x_bcd.unsqueeze(1)
        x_bnck = x_b1cd @ self.proj_1ndk
        metrics_1nkk = (self.coef_1nkk + self.coef_1nkk.transpose(-1, -2)) / 2
        scores_bncc = x_bnck @ metrics_1nkk @ x_bnck.transpose(-1, -2)
        scores_bncc = F.softmax(scores_bncc, dim=-1)
        x_bnck = scores_bncc @ x_bnck
        x_bcd = x_bnck.reshape(x_bcd.shape)
        x_bcd = x_bcd @ self.mixer_dd
        return x_bcd
