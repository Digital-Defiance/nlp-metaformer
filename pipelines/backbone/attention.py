

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



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class AttentionConfig(BaseModel):
    number_of_heads: int = 4
    kind: Literal["quadratic", "metric", "identity",  "scaled_dot_product"] = "quadratic"
    norm_input: Literal["LayerNorm"] | None = "LayerNorm"
    residual: bool = True
    masked: bool = True

    def to_model(self, embedd_config: EmbeddingsConfig):
        return Attention(self, embedd_config)


class Identity(nn.Module):
    def forward(self, x):
        return x

class Attention(nn.Module):
    
    def __init__(self, cfg: AttentionConfig, embedd_cfg: EmbeddingsConfig):
        super().__init__()
        
        if cfg.kind == "quadratic":
            self.attention = QuadraticAttention(cfg, embedd_cfg)
        elif cfg.kind == "metric":
            self.attention = MetricAttention(cfg, embedd_cfg)
        elif cfg.kind == "identity":
            self.attention = Identity()
        else:
            self.attention = CausalSelfAttention(cfg, embedd_cfg)

        if cfg.norm_input is not None:
            self.attention = nn.Sequential(
                nn.LayerNorm(embedd_cfg.dim),
                self.attention
            )
            
        if cfg.residual:
            self.attention = Residual(self.attention)
            
    def forward(self, x_bcd):
        return self.attention(x_bcd)
        
MINUS_INF = float("-inf")
class QuadraticAttention(nn.Module):
    def __init__(self, config: AttentionConfig, embedd_config: EmbeddingsConfig):
        super().__init__()
        k = embedd_config.dim // config.number_of_heads
        self.coef_1nkk = make_linear(1, config.number_of_heads, k, k).to("cuda")
        self.proj_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k).to("cuda")
        self.mixer_dd = make_linear(embedd_config.dim, embedd_config.dim).to("cuda")
        self.out_proj_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k).to("cuda")
        self.register_buffer("bias", torch.tril(torch.ones(embedd_config.ctx_win, embedd_config.ctx_win))
                                        .view(1, 1, embedd_config.ctx_win, embedd_config.ctx_win))

    def forward(self, x_bcd: Tensor):
        B, T, C = x_bcd.size() # batch size, sequence length, embedding dimensionality (n_embd)

        x_b1cd = x_bcd.unsqueeze(1)
        x_bnck = x_b1cd @ self.proj_1ndk
        scores_bncc = x_bnck @ self.coef_1nkk @ x_bnck.transpose(-1, -2)
        scores_bncc = scores_bncc *  (1.0 / math.sqrt(x_bnck.size(-1)))
        scores_bncc = scores_bncc.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        scores_bncc = F.softmax(scores_bncc, dim=-1)
        x_bnck = scores_bncc @ x_bnck
        x_bcnk = x_bnck.transpose(1, 2).contiguous()
        x_bcd = x_bcnk.view(*x_bcd.shape)
        x_bcd = x_bcd @ self.mixer_dd
        return x_bcd


class MetricAttention(nn.Module):
    def __init__(self, config: AttentionConfig, embedd_config: EmbeddingsConfig):
        super().__init__()
        k = embedd_config.dim // config.number_of_heads
        self.coef_1nkk = make_linear(1, config.number_of_heads, k, k).to("cuda")
        self.proj_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k).to("cuda")
        self.out_proj_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k).to("cuda")
        self.mixer_dd = make_linear(embedd_config.dim, embedd_config.dim).to("cuda")

    def forward(self, x_bcd: Tensor):
        x_b1cd = x_bcd.unsqueeze(1)
        x_bnck = x_b1cd @ self.proj_1ndk
        metrics_1nkk = (self.coef_1nkk + self.coef_1nkk.transpose(-1, -2)) / 2
        scores_bncc = x_bnck @ metrics_1nkk @ x_bnck.transpose(-1, -2)
        mask = torch.triu(torch.ones_like(scores_bncc), diagonal=1).bool()
        scores_bncc = scores_bncc.masked_fill(mask, MINUS_INF)
        scores_bncc = F.softmax(scores_bncc, dim=-1)
        x_bnck = scores_bncc @ ( x_b1cd @ self.out_proj_1ndk )
        x_bcnk = x_bnck.transpose(1, 2).contiguous()
        x_bcd = x_bcnk.view(*x_bcd.shape)
        x_bcd = x_bcd @ self.mixer_dd
        return x_bcd



class ScaledDotProductAttention(nn.Module):
    def __init__(self, config: AttentionConfig, embedd_config: EmbeddingsConfig):
        super().__init__()
        k = embedd_config.dim // config.number_of_heads

        self.n_embd = embedd_config.dim
        self.n_head = config.number_of_heads

        self.Wq_1ndk = make_linear( 1, config.number_of_heads, embedd_config.dim, k).to("cuda")

        self.Wk_1ndk = make_linear(1, config.number_of_heads, embedd_config.dim, k).to("cuda")

        self.Wv_1ndk = make_linear( 1, config.number_of_heads, embedd_config.dim, k).to("cuda")
        self.c_proj = nn.Linear(embedd_config.dim, embedd_config.dim, bias=False)
        


        self.mixer_1dd = make_linear(1, embedd_config.dim, embedd_config.dim).to("cuda")
        
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(embedd_config.ctx_win, embedd_config.ctx_win))
                                        .view(1, 1, embedd_config.ctx_win, embedd_config.ctx_win))


    def forward(self, x_bcd: Tensor):
        B, T, C = x_bcd.size() 

        x_b1cd = x_bcd.unsqueeze(1)
        q_bnck = x_b1cd @ self.Wq_1ndk
        k_bnck = x_b1cd @ self.Wk_1ndk
        v_bnck = x_b1cd @ self.Wv_1ndk

        scores_bncc = q_bnck @ k_bnck.transpose(-1, -2) 
        scores_bncc = scores_bncc *  (1.0 / math.sqrt(v_bnck.size(-1)))
        scores_bncc = scores_bncc.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        scores_bncc = F.softmax(scores_bncc, dim=-1)

        x_bnck = scores_bncc @ v_bnck
        x_bcnk = x_bnck.transpose(1, 2).contiguous()
        x_bcd = x_bcnk.view(B, T, C)
        x_bcd = self.c_proj(x_bcd)
        return x_bcd




class CausalSelfAttention(nn.Module):

    def __init__(self, config: AttentionConfig, embedd_config: EmbeddingsConfig):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embedd_config.dim, 3 * embedd_config.dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(embedd_config.dim, embedd_config.dim, bias=False)
        # regularization

        self.n_head = config.number_of_heads
        self.n_embd = embedd_config.dim
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(embedd_config.ctx_win, embedd_config.ctx_win))
                                        .view(1, 1, embedd_config.ctx_win, embedd_config.ctx_win))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = (  self.c_attn(x) ) .split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
  

        if not self.flash:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
