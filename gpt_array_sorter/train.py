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
load_dotenv()

class MLFlowSettings(BaseSettings):
    EXPERIMENT_ID: int
    TRACKING_URL: str
    LOG_SYSTEM_METRICS: bool = True

    @classmethod
    @contextmanager
    def start_run(cls):
        self = cls()
        mlflow.set_tracking_uri(self.TRACKING_URL)
        mlflow.enable_system_metrics_logging()
        with mlflow.start_run(
            experiment_id=self.EXPERIMENT_ID,
            log_system_metrics=self.LOG_SYSTEM_METRICS,
        ) as run:
            yield run



class ModelParameters(BaseSettings):
    """
    Represents the parameters of a model.

    Attributes:
        coordinates (int): The dimension of a vector embedding.
        tokens (int): The number of tokens in the vocabulary.
        words (int): The maximum number of words in a sentence (context window).
        number_of_blocks (int): The number of blocks in the model.
    """
    coordinates: int = 3*3
    tokens: int = 3
    words: int = 11
    number_of_blocks: int = 3

    class Config:
        env_file = ".env.model"

    @classmethod
    @contextmanager
    def make_model(cls):
        params = cls()
        nanoGPT = NanoGPT(params)
        nanoGPT.params = params

        mlflow.log_param("number_of_blocks", params.number_of_blocks)
        mlflow.log_param("coordinates", params.coordinates)
        mlflow.log_param("tokens", params.tokens)
        mlflow.log_param("words", params.words)
        mlflow.log_param("parameters", nanoGPT.count_parameters())
        mlflow.log_param("device", DEVICE)

        yield  nanoGPT.to(DEVICE)

        


class TrainConfiguration(BaseSettings):
    NUMBER_OF_EPOCHS: int = 100
    NUMBER_OF_BATCHES: int = 100
    LEARNING_RATE: float = 0.0005
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

torch.autograd.set_detect_anomaly(True)

with MLFlowSettings.start_run() as run:
    with ModelParameters.make_model() as nanoGPT:
        with TrainConfiguration.train(nanoGPT) as trainer:
            loss_function = trainer.create_loss_function()
            optimizer = trainer.create_optimizer()
            for epoch in range(trainer.NUMBER_OF_EPOCHS):                
                model = trainer.get_model()
                progress_bar = trainer.create_progress_bar(epoch)
                for batch, targets in progress_bar:
                    optimizer.zero_grad()
                    outputs = nanoGPT(batch)
                    loss = loss_function(outputs.transpose(-1, -2), targets)
                    loss.backward()
                    optimizer.step()
                    progress_bar.set_postfix(loss=loss.item())

                mlflow.log_metric("loss", loss.item(), epoch)
                mlflow.pytorch.log_model(nanoGPT, f"gpt_array_sorter_epoch_{epoch}")
    
 