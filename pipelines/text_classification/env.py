from constants import DEV_RUST_BINARY, DEFAULT_ATTENTION_MECHANISM
from pydantic_settings import BaseSettings
from typing import Literal
import os

from constants import SourceExecutable, AttentionMechanisms


from pydantic import  model_validator
class Data(BaseSettings):
    train_source: str = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/train.parquet"
    test_source: str = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/test.parquet"
    slices: int = 10
    batch_size: int = 32

class TrainingProcess(BaseSettings):
    use_gpu: bool = True
    executable_source: SourceExecutable = DEV_RUST_BINARY

  
class Train(BaseSettings):
    epochs: int = 1
    learning_rate: float = 1e-4


class Model(BaseSettings):
    encoding: Literal["tiktoken-gpt2"] = "tiktoken-gpt2"
    attention_kind: AttentionMechanisms = DEFAULT_ATTENTION_MECHANISM
    dimension: int = 32
    depth: int = 1
    heads: int = 2
    context_window: int = 300
    input_vocabolary: int = 60_000
    output_vocabolary: int = 5
    kernel_size: int | None = None


    @model_validator(mode='after')
    def check_kernel(self) -> 'Model':
        if self.attention_kind == "avg_pooling":
            assert self.kernel_size is not None
        return self

class MLFLowSettings(BaseSettings):
    mlflow_tracking_uri: str
    mlflow_tracking_username: str
    mlflow_tracking_password: str
    mlflow_run_id: str
    experiment_id: int = 1
    run_name: str | None = None


class Settings(BaseSettings):
    process: TrainingProcess
    model: Model 
    train: Train
    data: Data
    mlflow: MLFLowSettings

    @classmethod
    def from_env(cls):
        return cls(
            process = TrainingProcess(),
            model = Model(),
            train = Train(),
            data = Data(),
            mlflow = MLFLowSettings(),
        )

    def yield_flattened_items(self, node: dict | None = None):
        node: dict[str, any] = node or self.model_dump()

        for key, value in node.items():
            if value is None:
                continue

            if not isinstance(value, dict):
                yield (key.upper(), str(value))
                continue

            yield from self.yield_flattened_items(node = value)


    def to_env(self) -> None:
        """ Loads every key value pair to the environment. """

        for key, value in self.yield_flattened_items():
            os.environ[key] = value


