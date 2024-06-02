"""

"""

from pipelines.next_token.model import TransformerConfig
import numpy as np
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import os
import requests
import tiktoken
import numpy as np
from torch import Tensor
from prefect import flow, task
from torch import nn
from pipelines.commons import make_linear
from pipelines.next_token.model import NextTokenPredictor, Scheduler
from pipelines.backbone.attention import AttentionConfig
from pipelines.backbone.feedforward import FeedForwardConfig
from pydantic import BaeModel
from typing import Literal
from pipelines.commons import DEVICE

class DataLoader(BaseModel):
    iterations: int
    ctx_win: int | None = None
    datas_path: str
    batch_size: int

    def set_ctx_win(self, val):
        self.ctx_win = val

    def __len__(self):
        return self.size

    def __iter__(self):
        for _ in range(self.size):
            data = np.memmap(data_path, dtype=np.uint16, mode='r')
            idx_b = torch.randint(len(data) - block_size, (batch_size,))
            x_bcd = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in idx_b])
            y_bcd = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in idx_b])
            yield x_bcd.pin_memory().to(DEVICE, non_blocking=True), y_bcd.pin_memory().to(DEVICE, non_blocking=True)

@flow
def train_shakespear(
    cfg: TransformerConfig = TransformerConfig(
        embeddings_cfg = dict(
            ctx_win = 256, 
            dim= 384,
        ),
        body = 6*[
            AttentionConfig(
                kind="scaled_dot_product",
                number_of_heads = 6,
                masked = True,
                norm_input = "LayerNorm",
                residual = True,
            ),
            FeedForwardConfig(
                scale = 4,
                activation = "gelu",
                norm_input = "LayerNorm",
                residual = True,
            )
        ]
    ),
    val_dataset: DataLoader = DataLoader(
        iterations = 1,
        datas_path = os.path.join(os.path.dirname(__file__), 'val.bin'),
        batch_size = 32,
    ),
    train_dataset:  DataLoader = DataLoader(
        iterations = 100,
        datas_path = os.path.join(os.path.dirname(__file__), 'train.bin'),
        batch_size = 32,
    ),
    scheduler: Scheduler = Scheduler(
        warmup_iters = 100,
        lr_decay_iters = 5000,
        learning_rate = 1e-3,
        min_lr = 1e-4,
    ),
    dataset_url: str = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    txt_path: str = os.path.join(os.path.dirname(__file__), 'input.txt'),
    max_epochs: int = 1000,
    barebones: bool = True,
):

    if not os.path.exists(txt_path):
        with open(txt_path, 'w', encoding='utf-8') as f:
            response = requests.get(DATASET_URL)
            f.write(response.text)
            
    train_dataset_missing = not os.path.exists(train_dataset.datas_path)
    test_dataset_missing = not os.path.exists(val_dataset.datas_path)
    if train_dataset_missing or test_dataset_missing:        
        with open(TXT_PATH, 'r', encoding='utf-8') as f:
            data = f.read()

        n = len(data)
        enc = tiktoken.get_encoding("gpt2")

        if train_dataset_missing:
            train_c = data[:int(n*0.9)]
            train_q = enc.encode_ordinary(train_c)
            train_q = np.array(train_q, dtype=np.uint16)
            train_q.tofile(train_dataset.datas_path)
            del train_q, train_c
            
        if test_dataset_missing:
            val_c = data[int(n*0.9):]
            val_q = enc.encode_ordinary(val_c)
            val_q = np.array(val_q, dtype=np.uint16)
            val_q.tofile(VAL_PATH)
            del val_c, val_q
    
    train_dataset.set_ctx_win(cfg.embeddings_cfg.ctx_win)
    val_dataset.set_ctx_win(cfg.embeddings_cfg.ctx_win)

    class ShakeSpeareData(LightningDataModule):
        def train_dataloader(self):
            return train_dataset
        def val_dataloader(self):
            return val_dataset

    model = NextTokenPredictor(cfg).to(DEVICE)
    # print("compiling model")
    # model = torch.compile(model)
    print(model)
    trainer = Trainer(max_epochs=max_epochs, barebones=barebones)
    trainer.fit(model, ShakeSpeareData())

if __name__ == "__main__":
    train_shakespear()