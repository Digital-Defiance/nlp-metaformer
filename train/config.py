
from pydantic_settings import BaseSettings
import mlflow
from dotenv import load_dotenv
from typing import  Optional
import mlflow
import tiktoken



gpt2_encoder = tiktoken.get_encoding("gpt2")


class MLFlowSettings(BaseSettings):
    experiment_id: int
    run_id: Optional[str] = None
    tracking_uri: str
    tracking_username: str
    tracking_password: str
    log_system_metrics: bool = True

    class Config:
        env_prefix = "MLFLOW_"


class TrainConfiguration(BaseSettings):
    number_of_epochs: int = 100
    number_of_batches: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    loss_function: str = "CrossEntropyLoss"

    def save_to_mlflow(self):
        mlflow.log_param("epochs", self.number_of_epochs)
        mlflow.log_param("batches", self.number_of_batches)
        mlflow.log_param("batch_size", self.batch_size)
        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("loss_function", self.loss_function)

    @classmethod
    def load_from_mlflow(cls) -> "TrainConfiguration":
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        return cls(
            number_of_epochs= run.data.params.get("epochs", None),
            number_of_batches= run.data.params.get("batches", None),
            batch_size= run.data.params.get("batch_size", None),
            learning_rate= run.data.params.get("learning_rate", None),
            loss_function= run.data.params.get("loss_function", None),
        )

