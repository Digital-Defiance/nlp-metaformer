
import mlflow.pytorch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Literal
import mlflow
import torch
from mlflow.entities import RunStatus

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PauseRunException(Exception):
    pass

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
    def continue_or_create_run(cls, run_id: str | None):
        try:
            if run_id is None:
                with cls.start_run() as mlflow_handler:
                    yield mlflow_handler
                return
            self = cls()
            self._run_id = run_id
            mlflow.set_tracking_uri(self.TRACKING_URL)
            mlflow.enable_system_metrics_logging()
            with mlflow.start_run(
                run_id=run_id,
                log_system_metrics=self.LOG_SYSTEM_METRICS,
            ):
                yield self
        except PauseRunException:
            client = mlflow.tracking.MlflowClient()
            status = RunStatus.SCHEDULED
            client.set_terminated(run_id, status=RunStatus.to_string(status))


    def get_status(self) -> Literal["RUNNING", "FINISHED", "FAILED", "SCHEDULED"]:
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

    def save_to_mlflow(self):
        mlflow.log_param("epochs", self.number_of_epochs)
        mlflow.log_param("batches", self.number_of_batches)
        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("loss_function", self.loss_function)

    @classmethod
    def load_from_mlflow(cls) -> "TrainConfiguration":
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        return cls(
            number_of_epochs= run.data.params.get("epochs", None),
            number_of_batches= run.data.params.get("batches", None),
            learning_rate= run.data.params.get("learning_rate", None),
            loss_function= run.data.params.get("loss_function", None),
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

    def save_to_mlflow(self):
        mlflow.log_param("number_of_blocks", self.number_of_blocks)
        mlflow.log_param("coordinates", self.coordinates)
        mlflow.log_param("tokens", self.tokens)
        mlflow.log_param("words", self.words)
    
    @classmethod
    def load_from_mlflow(cls) -> "ModelHandler":
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        return cls(
            number_of_blocks= run.data.params.get("number_of_blocks", None),
            coordinates= run.data.params.get("coordinates", None),
            tokens= run.data.params.get("tokens", None),
            words= run.data.params.get("words", None),
        )
