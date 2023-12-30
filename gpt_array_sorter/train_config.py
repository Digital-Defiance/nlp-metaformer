
import mlflow.pytorch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from contextlib import contextmanager
import os
from typing import Literal

load_dotenv()

class MLFlowHandler(BaseSettings):
    EXPERIMENT_ID: int
    TRACKING_URL: str
    LOG_SYSTEM_METRICS: bool = True

    _run_id: str = None

    @classmethod
    @contextmanager
    def start_run(cls):
        self = cls()
        mlflow.set_tracking_uri(self.TRACKING_URL)

        with mlflow.start_run(experiment_id=self.EXPERIMENT_ID) as run:
            self._run_id = run.info.run_id
            yield self

    @classmethod
    @contextmanager
    def continue_run(cls):
        self = cls()
        mlflow.set_tracking_uri(self.TRACKING_URL)
        mlflow.enable_system_metrics_logging()
        with mlflow.start_run(
            run_id=os.environ.get("RUN_ID"),
            log_system_metrics=self.LOG_SYSTEM_METRICS,
        ) as run:
            self._run_id = run.info.run_id
            yield self


    def get_status(self) -> Literal["RUNNING", "FINISHED", "FAILED"]:
        run = mlflow.get_run(self._run_id)
        return run.info.status

    def get_parameter(self, key):
        run = mlflow.get_run(self._run_id)
        return run.data.params.get(key, None)

class TrainConfiguration(BaseSettings):
    number_of_epochs: int = 10
    number_of_batches: int = 50
    learning_rate: float = 0.001
    loss_function: str = "CrossEntropyLoss"

    def save_to_mlflow(self, mlflow_handler):
        mlflow_handler.log_param("epochs", self.NUMBER_OF_EPOCHS)
        mlflow_handler.log_param("batches", self.NUMBER_OF_BATCHES)
        mlflow_handler.log_param("learning_rate", self.LEARNING_RATE)
        mlflow_handler.log_param("loss_function", self.LOSS_FUNCTION)

    @classmethod
    def load_from_mlflow(cls, mlflow_handler):
        return cls(
            NUMBER_OF_EPOCHS=mlflow_handler.get_parameter("epochs"),
            NUMBER_OF_BATCHES=mlflow_handler.get_parameter("batches"),
            LEARNING_RATE=mlflow_handler.get_parameter("learning_rate"),
            LOSS_FUNCTION=mlflow_handler.get_parameter("loss_function"),
        )

class ModelHandler(BaseSettings):
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

    def save_to_mlflow(self, mlflow_handler):
        mlflow_handler.log_param("number_of_blocks", self.number_of_blocks)
        mlflow_handler.log_param("coordinates", self.coordinates)
        mlflow_handler.log_param("tokens", self.tokens)
        mlflow_handler.log_param("words", self.words)
    
    @classmethod
    def load_from_mlflow(cls, mlflow_handler):
        return cls(
            number_of_blocks=mlflow_handler.get_parameter("number_of_blocks"),
            coordinates=mlflow_handler.get_parameter("coordinates"),
            tokens=mlflow_handler.get_parameter("tokens"),
            words=mlflow_handler.get_parameter("words"),
        )
