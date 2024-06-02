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
from pipelines.next_token.model import NextTokenPredictor
from pipelines.backbone.attention import AttentionConfig
from pipelines.backbone.feedforward import FeedForwardConfig


from typing import Literal

DATASET_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
TXT_PATH = os.path.join(os.path.dirname(__file__), 'input.txt')
TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'train.bin')
VAL_PATH = os.path.join(os.path.dirname(__file__), 'val.bin')
DEVICE = "cuda"










class ShakeSpeareData(LightningDataModule):

    def __init__(self, ctx_win):
        super().__init__()
        self.ctx_win = ctx_win

    def data_loader(self, split: Literal["train", "val"]):
        data_path = TRAIN_PATH if split == "train" else VAL_PATH
        iterations = 10000 if split == "train" else 1
        block_size = self.ctx_win 
        batch_size = 32

        class DataLoader:

            def __len__(self):
                return iterations

            def __iter__(self):
                for _ in range(iterations):
                    data = np.memmap(data_path, dtype=np.uint16, mode='r')
                    idx_b = torch.randint(len(data) - block_size, (batch_size,))
                    x_bcd = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in idx_b])
                    y_bcd = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in idx_b])
                    yield x_bcd.pin_memory().to(DEVICE, non_blocking=True), y_bcd.pin_memory().to(DEVICE, non_blocking=True)
    
        return DataLoader()

    def train_dataloader(self):
        return self.data_loader('train')

    def val_dataloader(self):
        return self.data_loader('val')

@task
def ensure_dataset():
    if not os.path.exists(TXT_PATH):
        with open(TXT_PATH, 'w', encoding='utf-8') as f:
            response = requests.get(DATASET_URL)
            f.write(response.text)

    if os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH):
        return
    
    
    with open(TXT_PATH, 'r', encoding='utf-8') as f:
        data = f.read()

    n = len(data)
    enc = tiktoken.get_encoding("gpt2")

    if not os.path.exists(TRAIN_PATH):
        train_data = data[:int(n*0.9)]
        train_ids = enc.encode_ordinary(train_data)
        train_ids = np.array(train_ids, dtype=np.uint16)
        train_ids.tofile(TRAIN_PATH)
        
    if not os.path.exists(VAL_PATH):
        val_data = data[int(n*0.9):]
        val_ids = enc.encode_ordinary(val_data)
        val_ids = np.array(val_ids, dtype=np.uint16)
        val_ids.tofile(VAL_PATH)


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
            ),
            FeedForwardConfig(
                scale = 4,
            )
        ]
    )
):
    
    ensure_dataset()
    
    data_module = ShakeSpeareData(cfg.embeddings_cfg.ctx_win)
    model = NextTokenPredictor(cfg).to("cuda")
    # print("compiling model")
    # model = torch.compile(model)
    print(model)
    trainer = Trainer(max_epochs=1000, barebones=True)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    train_shakespear()