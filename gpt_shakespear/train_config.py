
from pydantic_settings import BaseSettings
import mlflow
from dotenv import load_dotenv
from typing import  Optional
import mlflow

load_dotenv()

class MLFlowSettings(BaseSettings):
    experiment_id: int
    run_id: Optional[str] = None
    tracking_url: str
    tracking_username: str
    tracking_password: str
    log_system_metrics: bool = True

    class Config:
        env_prefix = "MLFLOW_"


class TrainConfiguration(BaseSettings):
    number_of_epochs: int = 20
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
