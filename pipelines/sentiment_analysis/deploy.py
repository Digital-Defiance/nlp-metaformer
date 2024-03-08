import os
import asyncio
import subprocess
from pathlib import Path
from contextlib import contextmanager
from typing import Literal

import duckdb
from duckdb.typing import *
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from pydantic_settings import BaseSettings
from prefect import flow, get_run_logger, task
from prefect.runner.storage import GitRepository
from safetensors import torch as stt
import tiktoken

SAVE_PATH: str = "output.safetensors"

PathStr = str

DEV_RUST_BINARY: str = "/__w/llm-voice-chat/llm-voice-chat/target/debug/llm-voice-chat"

class Data(BaseSettings):
    source: str = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet"
    slices: int = 1
    batch_size: int = 32

    def to_cmd_args(self) -> str:
        return f"--batch-size {self.batch_size}"

class TrainingProcess(BaseSettings):
    use_gpu: bool = True
    executable_source: Literal[
        "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/v0.0.2/llm-voice-chat",
        "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/v0.0.1/llm-voice-chat",
        DEV_RUST_BINARY,
    ] = DEV_RUST_BINARY

    def to_cmd_args(self) -> str:
        return "--use-gpu" if self.use_gpu else ""



class Train(BaseSettings):
    epochs: int = 1
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

    def to_cmd_args(self):
        args = [f"--{key.replace('_', '-')} {value}" for key, value in self.model_dump().items()]
        return " ".join(args)


class Settings(BaseSettings):
    process: TrainingProcess
    model: Model 
    train: Train
    data: Data

    def to_cmd_args(self) -> str:
        args = self.model.to_cmd_args()
        args += " " + self.train.to_cmd_args()
        args += " " + self.data.to_cmd_args()
        args += " " + self.process.to_cmd_args()
        return args

ENCODER = tiktoken.get_encoding("gpt2")

def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text.lower())

def set_seed(conn: duckdb.DuckDBPyConnection, seed: float) -> None:
    conn.execute(f"SELECT setseed(?) as ign;", [seed])

def create_table(conn: duckdb.DuckDBPyConnection, dataset_link: str):
    conn.execute(f"""
        CREATE TABLE dataset AS
        SELECT id
        FROM '{dataset_link}';
    """)

def add_epoch_column(conn: duckdb.DuckDBPyConnection, epoch_idx: int, number_of_partions: int):
    conn.execute(f"""
        ALTER TABLE dataset
        ADD COLUMN epoch_{epoch_idx}
        INTEGER DEFAULT trunc( {number_of_partions}*random() );
    """)

def select_partition(conn: duckdb.DuckDBPyConnection, dataset_link: str, epoch_idx: int, slice_idx: int):
    return conn.execute(f"""
        SELECT sentiment, review
        FROM '{dataset_link}' as remote
        JOIN dataset ON (dataset.id = remote.id)
        WHERE dataset.epoch_{epoch_idx}={slice_idx};
    """)

@contextmanager
def dataset_partitioning(number_of_epochs, number_of_partions, dataset_link: str, seed = 0.5):
    with duckdb.connect() as conn:
        conn: duckdb.DuckDBPyConnection
        set_seed(conn, seed)
        create_table(conn, dataset_link)

        # note: sampling is pre-computed here, data is fetched on demand with "fetch_data"
        for epoch_idx in range(number_of_epochs):
            add_epoch_column(conn, epoch_idx, number_of_partions)
        
        def fetch_data(epoch_idx: int, slice_idx: int) -> tuple[list[int], list[str]]:
            sentiments, reviews = [], []
            for sentiment, text in select_partition(conn, dataset_link, epoch_idx, slice_idx).fetchall():
                sentiment = 1 if sentiment == "Pos" else 0
                sentiments.append(sentiment)
                reviews.append(text)
            return sentiments, reviews
        yield fetch_data






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
async def run_rust_binary(path_to_rust_binary: str, arguments: str):
    logger = get_run_logger()
    cmd = f'{path_to_rust_binary} {arguments} --path-to-slice {SAVE_PATH}'
    logger.info(f"Running command: {cmd}")
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
        while (code := process.poll()) is None:
            if (output := process.stdout.readline()):
                logger.info(output.strip())
            await asyncio.sleep(10)

        if code != 0:
            error_msg = ""
            if process.stderr:
                error_msg = process.stderr.read()
                logger.error(error_msg)
            raise RuntimeError(f"Rust program exited with non-zero code {code}: {error_msg}")

@task
async def make_rust_executable(path_to_rust_binary: str) -> None:
    logger = get_run_logger()
    cmd = f'chmod +x {path_to_rust_binary}'
    logger.info(f"Running command: {cmd}")
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
        while (code := process.poll()) is None:
            if (output := process.stdout.readline()):
                logger.info(output.strip())
            await asyncio.sleep(1)

        if code != 0:
            error_msg = ""
            if process.stderr:
                error_msg = process.stderr.read()
                logger.error(error_msg)
            raise RuntimeError(f"Command exited with non-zero code {code}: {error_msg}")

@flow
async def training_loop(settings: Settings):

    path_to_rust_binary = DEV_RUST_BINARY
    if settings.process.executable_source != DEV_RUST_BINARY:
        path_to_rust_binary = download_rust_binary(settings.process.executable_source)
        await make_rust_executable(path_to_rust_binary)

    await run_rust_binary(path_to_rust_binary, settings.to_cmd_args())

@task
def clean_safetensor_files():
    if Path(SAVE_PATH).exists():
        os.remove(SAVE_PATH)

@task
async def wait_data_consumption():
    while Path(SAVE_PATH).exists():
        await asyncio.sleep(1)

@task
def fetch_and_preprocess_data(fetch_data: callable, epoch: int, slice: int, context_window: int):
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
def save_data(data):
    stt.save_file(data, SAVE_PATH)

@flow
async def data_worker(settings: Settings):
    logger = get_run_logger()
    logger.info("Partitioning dataset.")
    with dataset_partitioning(
        number_of_epochs = settings.train.epochs,
        number_of_partions = settings.data.slices,
        dataset_link = settings.data.source
    ) as fetch_data:
        clean_safetensor_files()
        for epoch in range(settings.train.epochs):
            for slice in range(settings.data.slices):
                logger.info(f"Constructing slice {slice} for epoch {epoch}")
                data = fetch_and_preprocess_data(fetch_data, epoch, slice, settings.model.context_window)
                await wait_data_consumption()
                save_data(data)


@flow
async def main(settings: Settings):
    parallel_subflows = [training_loop(settings), data_worker(settings)]
    await asyncio.gather(*parallel_subflows)


class EnvironmentSettings(BaseSettings): 
    LLMVC_ENVIRONMENT: str = "prod"
    github_url: str = "https://github.com/Digital-Defiance/llm-voice-chat.git"
    entrypoint: str  = "pipelines/sentiment_analysis/deploy.py:main"
    name: str = "sentiment-analysis"
    workpool: str = "spot-hybrid"

    @staticmethod
    def get_active_branch_name():
        head_dir = Path(".") / ".git" / "HEAD"
        with head_dir.open("r") as f: content = f.read().splitlines()
        for line in content:
            if line[0:4] == "ref:":
                return line.partition("refs/heads/")[2]

if __name__ == "__main__":
    settings = EnvironmentSettings()

    if settings.LLMVC_ENVIRONMENT == "dev":
        main.serve(name = settings.name + "-test")
    else:
        git_repo = GitRepository(url = settings.github_url, branch = settings.get_active_branch_name())
        main_flow = main.from_source(entrypoint = settings.entrypoint, source = git_repo,)
        main_flow.deploy(name = settings.name, work_pool_name = settings.workpool)
