

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pipelines.backbone.transformer import TransformerConfig
from torch import nn, Tensor
from pipelines.commons import make_linear
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torch
import math
from pipelines.commons import DEVICE

class Scheduler(BaseModel):
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4

    def get_lr(self, it):
        # 1) Linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        lr =  self.min_lr + coeff * (self.learning_rate - self.min_lr)
        return lr


class NextTokenPredictor(LightningModule):
    def __init__(self, model_cfg: TransformerConfig, scheduler: Scheduler):
        super().__init__()
        self.backbone = model_cfg.to_model()
        self.layernorm = nn.LayerNorm(model_cfg.embeddings_cfg.dim)
        
        self.proj_dt = self.backbone.embedder.vocabolary.weight  #  make_linear(model_cfg.embeddings_cfg.dim, model_cfg.embeddings_cfg.vocab_size)
        self.scheduler = scheduler

    def generate_text(self, text: str):
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        x_bc = enc.encode_ordinary(text)
        x_bc = np.array(x_bc, dtype=np.uint16)

        for _ in range(10):
            x_bct = self(x_bc)
            token = torch.argmax(x_bct, dim=-1)

    def forward(self, x_bc: Tensor) -> Tensor:
        x_bcd = self.backbone(x_bc)
        x_bcd = self.layernorm(x_bcd)
        logits_bct = x_bcd @ self.proj_dt
        return logits_bct

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        iteration = 100*(self.trainer.current_epoch ) + batch_idx
        lr = self.scheduler.get_lr(iteration)
        optimizer = self.optimizers()
        for g in optimizer.param_groups:
            g['lr'] = lr
        self.log("lr",lr, prog_bar=True)
        x_bc, y_bc = batch
        x_bc, y_bc = x_bc.to("cuda"), y_bc.to("cuda")
        logits_bct = self(x_bc)
        logits_btc = logits_bct.transpose(-1, -2)
        loss = F.cross_entropy(logits_btc, y_bc)
        self.log("train/loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x_bc, y_bc = batch
        logits_bct = self(x_bc)
        logits_btc = logits_bct.transpose(-1, -2)
        loss = F.cross_entropy(logits_btc, y_bc)
        preds_bct = torch.argmax(logits_bct, dim=-1)
        correct = (preds_bct == y_bc).float()
        accuracy = correct.sum() / correct.numel()
        self.log("val/loss", loss.item(), prog_bar=True)
        self.log("val/acc", 100*accuracy.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), betas=(0.95, 0.99), lr=1e-3)
