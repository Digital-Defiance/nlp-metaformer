
from pydantic import BaseModel
from torch import nn
from pipelines.backbone.embedder import EmbeddingsConfig
from pipelines.backbone.residual import Residual
from typing import Literal
from torch import Tensor

class FeedForwardConfig(BaseModel):
    scale: int = 3
    activation: Literal["gelu"] = "gelu"
    norm_input: Literal["LayerNorm"] | None = "LayerNorm"
    residual: bool = True
    
    def to_model(self, embedd_config: EmbeddingsConfig):
        return FeedForward(self, embedd_config)

class FeedForward(nn.Module):
    def __init__(self, cfg: FeedForwardConfig, embedd_cfg: EmbeddingsConfig):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(embedd_cfg.dim, cfg.scale*embedd_cfg.dim, bias = True),
            nn.GELU(),
            nn.Linear(cfg.scale*embedd_cfg.dim, embedd_cfg.dim, bias = True)
        )

        if cfg.norm_input is not None:
            self.ff = nn.Sequential(
                nn.LayerNorm(embedd_cfg.dim),
                self.ff
            )
            
        if cfg.residual:
            self.ff = Residual(self.ff)

    def forward(self, x_bcd: Tensor) -> Tensor:
        return self.ff(x_bcd)


