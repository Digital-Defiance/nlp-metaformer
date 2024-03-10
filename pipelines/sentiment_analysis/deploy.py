"""

"""

import os
import subprocess
from pathlib import Path
from contextlib import contextmanager
from typing import Literal

import duckdb
from duckdb.typing import *
import mlflow
from pydantic_settings import BaseSettings
from prefect import flow, get_run_logger, task
import numpy as np
import tiktoken
from prefect_shell import ShellOperation

SAVE_PATH: str = "output.safetensors"
DEV_RUST_BINARY: str = "/__w/llm-voice-chat/llm-voice-chat/target/debug/llm-voice-chat"


class RustExitedWithError(RuntimeError):
    def __init__(self, code, error_msg):
        super().__init__(f"Command exited with non-zero code {code}: {error_msg}")

class Data(BaseSettings):
    
    train_source: str = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/train.parquet"
    test_source: str = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/test.parquet"
    slices: int = 1
    batch_size: int = 32


SourceExecutable = Literal[
    "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/v0.0.2/llm-voice-chat",
    "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/v0.0.1/llm-voice-chat",
    "/__w/llm-voice-chat/llm-voice-chat/target/debug/llm-voice-chat",
]

class TrainingProcess(BaseSettings):
    use_gpu: bool = False
    executable_source: SourceExecutable = DEV_RUST_BINARY

  

class Train(BaseSettings):
    epochs: int = 100
    learning_rate: float = 1e-4

    def to_cmd_args(self):
        return f"--learning-rate {self.learning_rate}"

class Model(BaseSettings):
    encoding: Literal["tiktoken-gpt2"] = "tiktoken-gpt2"
    attention_kind: Literal["Quadratic", "Metric", "ScaledDotProduct"] = "Quadratic"
    dimension: int = 32
    depth: int = 1
    heads: int = 2
    context_window: int = 300
    input_vocabolary: int = 60_000
    output_vocabolary: int = 5



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

            if not isinstance(value, dict):
                yield (key.upper(), str(value))
                continue

            yield from self.yield_flattened_items(node = value)


    def to_env(self) -> None:
        """ Loads every key value pair to the environment. """

        for key, value in self.yield_flattened_items():
            os.environ[key] = value





ENCODER = tiktoken.get_encoding("gpt2")

def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text.lower())



class DatasetConnection:

    def __init__(self, conn: duckdb.DuckDBPyConnection, dataset_link: str):
        self.dataset_link = dataset_link
        self.conn = conn

        self.conn.execute(f"""
            CREATE TABLE dataset AS
            SELECT id
            FROM '{self.dataset_link}';
        """)

        self.count = self.conn.execute("""SELECT COUNT(*) FROM dataset""").fetchall()[0][0]


    def generate_randomization(self, total_epochs: int, seed: int = 42):

        np.random.seed(seed)

        for epoch_idx in range(total_epochs):

            self.conn.execute(f"""
                ALTER TABLE dataset
                ADD COLUMN epoch_{epoch_idx}
                INTEGER;
            """)

            permutation = np.random.permutation(self.count) + 1
            values = ", ".join((f"({val},)" for val in permutation))
            self.conn.execute(f"""
                INSERT INTO dataset (epoch_{epoch_idx})
                VALUES {values};
            """)

    def select_partition(self, epoch_idx: int, start: int, limit: int):
        return self.conn.execute(f"""
            SELECT sentiment, review
            FROM '{self.dataset_link}' as remote
            JOIN dataset ON (dataset.epoch_{epoch_idx} = remote.id)
            OFFSET {start}
            LIMIT {limit};
        """)



@contextmanager
def dataset_partitioning(number_of_epochs, number_of_partions, dataset_link: str, seed: int = 42):
    with duckdb.connect() as conn:
        conn: duckdb.DuckDBPyConnection
        table = DatasetConnection(conn, dataset_link)
        table.generate_randomization(number_of_epochs, seed=seed)
        # res = table.select_partition(0, 1, 100).fetchall()
        
        def fetch_data(epoch_idx: int, slice_idx: int) -> tuple[list[int], list[str]]:
            sentiments, reviews = [], []
            limit = table.count // number_of_partions
            offset = limit*slice_idx
            for sentiment, text in table.select_partition(epoch_idx, offset, limit).fetchall():
                sentiment = 1 if sentiment == "Pos" else 0
                sentiments.append(sentiment)
                reviews.append(text)
            return sentiments, reviews
        yield fetch_data


@task
def fetch_and_preprocess_data(fetch_data: callable, epoch: int, slice: int, context_window: int):
    from torch.nn.utils.rnn import pad_sequence
    from torch import tensor


    raw_sentiments, raw_reviews = fetch_data(epoch, slice)
    sentiments = tensor(raw_sentiments)
    reviews = []
    for text in raw_reviews:
        text = encode_text(text)[0:context_window]
        text = tensor(text)
        reviews.append(text)

    return {
        'Y': sentiments,
        "X": pad_sequence(reviews, batch_first=True)
    }

@task
def save_data(idx: int, data) -> None:
    from safetensors import torch as stt
    stt.save_file(data, f"{idx}_" + SAVE_PATH)






































@task
def download_rust_binary(url: str) -> str:
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    save_path = "./train"
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    return save_path

@task
def make_rust_executable(path_to_rust_binary: str) -> None:
    logger = get_run_logger()
    cmd = f'chmod +x {path_to_rust_binary}'
    logger.info(f"Running command: {cmd}")
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
        while (code := process.poll()) is None:
            if (output := process.stdout.readline()):
                logger.info(output.strip())

        if code != 0:
            error_msg = ""
            if process.stderr:
                error_msg = process.stderr.read()
                logger.error(error_msg)
            raise RustExitedWithError(code, error_msg)


@task
def run_rust_binary(path_to_rust_binary: str):
    ShellOperation(
        commands=[path_to_rust_binary],
        env = { key: value for key, value in Settings.from_env().yield_flattened_items() }
    ).run()


@task
def write_training_slices():

    data = Data()
    train = Train()

    with dataset_partitioning(
        number_of_epochs = train.epochs,
        number_of_partions = data.slices,
        dataset_link = data.train_source
    ) as fetch_data:
        logger = get_run_logger()
        step = 0
        for epoch in range(train.epochs):
            for slice in range(data.slices):
                logger.info(f"Constructing slice {slice} for epoch {epoch}")
                processed_data = fetch_and_preprocess_data.fn(fetch_data, epoch, slice, Model().context_window)
                save_data.fn(step, processed_data)
                step += 1



@task
def log_params():
    mlflow.log_params({
        **TrainingProcess().model_dump(),
        **Model().model_dump(),
        **Train().model_dump(),
        **Data().model_dump(),
    })

@task
def prepare_validation_slice():
    train = Train()
    model = Model()
    data = Data()
    with dataset_partitioning(
        number_of_epochs = train.epochs,
        number_of_partions = data.slices,
        dataset_link = data.test_source
    ) as fetch_data:
        epoch = 0
        slice = 0
        data = fetch_and_preprocess_data.fn(fetch_data, epoch, slice, model.context_window)
        slice_idx = -1 # negative numbers for eval split
        save_data.fn(slice_idx, data)


@flow
def main(
    process: TrainingProcess,
    model: Model,
    train: Train,
    data: Data,
    experiment_id: int = 1,
    run_name: str | None = None,
):


    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
    
        Settings(
            process = process, 
            model = model,
            train = train,
            data = data,
            mlflow = MLFLowSettings(
                experiment_id=experiment_id,
                run_name=run_name,
                mlflow_run_id=run.info.run_id
            )
        ).to_env()
    
        log_params.submit()
        prepare_validation_slice.submit()
        write_training_slices.submit()

        path_to_rust_binary = DEV_RUST_BINARY
        if process.executable_source != DEV_RUST_BINARY:
            path_to_rust_binary = download_rust_binary(process.executable_source)
            make_rust_executable(path_to_rust_binary)

        training_loop = run_rust_binary.submit(path_to_rust_binary)
        training_loop.wait()




if __name__ == "__main__":

    class EnvironmentSettings(BaseSettings): 
        llmvc_environment: Literal["prod", "dev"] = "prod"

    main.serve(
        name = f"sentiment-analysis-{EnvironmentSettings().llmvc_environment}"
    )

