

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pipelines.backbone.transformer import TransformerConfig
from torch import nn, Tensor
from pipelines.commons import make_linear
import torch
import torch.nn.functional as F


class NextTokenPredictor(LightningModule):
    def __init__(self, model_cfg: TransformerConfig):
        super().__init__()
        self.backbone = model_cfg.to_model()
        self.layernorm = nn.LayerNorm(model_cfg.embeddings_cfg.dim)
        self.proj_1dt = make_linear(1, model_cfg.embeddings_cfg.dim, model_cfg.embeddings_cfg.vocab_size)

    def forward(self, x_bc: Tensor) -> Tensor:
        x_bcd = self.backbone(x_bc)
        x_bcd = self.layernorm(x_bcd)
        logits_bct = x_bcd @ self.proj_1dt
        return logits_bct

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x_bc, y_bc = batch
        logits_bct = self(x_bc)
        logits_btc = logits_bct.transpose(-1, -2)
        loss = F.cross_entropy(logits_btc, y_bc)
        print(loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x_bc, y_bc = batch
        logits_bct = self(x_bc)
        logits_btc = logits_bct.transpose(-1, -2)
        loss = F.cross_entropy(logits_btc, y_bc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)