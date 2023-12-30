from model import NanoGPT
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
from data import generate_data
from pydantic_settings import BaseSettings
from system_parameters import DEVICE
from dotenv import load_dotenv
from contextlib import contextmanager
from tqdm import tqdm
from typing import Any
from mlflow_handler import MLFlowHandler
load_dotenv()

class TrainConfiguration(BaseSettings):
    NUMBER_OF_EPOCHS: int = 10
    NUMBER_OF_BATCHES: int = 50
    LEARNING_RATE: float = 0.001
    LOSS_FUNCTION: str = "CrossEntropyLoss"

    model: Any = None

    def create_loss_function(self):
        if self.LOSS_FUNCTION == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()

    def create_progress_bar(self, epoch):
        return tqdm(generate_data(
            batches=self.NUMBER_OF_BATCHES,
            params=self.model.params
        ), desc=f"Epoch {epoch}", leave=True)
    
    def get_model(self):
        return self.model.train()
    
    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    class Config:
        env_file = ".env.train"

    @classmethod
    @contextmanager
    def train(cls, model):
        self = cls()
        mlflow.log_param("learning_rate", self.LEARNING_RATE)
        mlflow.log_param("loss_function", self.LOSS_FUNCTION)
        mlflow.log_param("number_of_epochs", self.NUMBER_OF_EPOCHS)
        mlflow.log_param("number_of_batches", self.NUMBER_OF_BATCHES)
        self.model = model
        yield self