import os
import asyncio
import subprocess
from pathlib import Path

import asyncio
from contextlib import contextmanager
from typing import Literal


import duckdb
from duckdb.typing import *

from torch import Tensor, tensor, save
from torch.nn.utils.rnn import pad_sequence

from pydantic_settings.sources import YamlConfigSettingsSource
from pydantic_settings import BaseSettings

from prefect import flow, serve, get_run_logger, task, variables
from prefect.runner.storage import GitRepository

from safetensors import torch as stt

import tiktoken

SAVE_PATH = "output.safetensors"

PathStr = str



class Data(BaseSettings):
    source: str = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet"
    slices: int = 5
    batch_size: int = 32

class TrainingProcess(BaseSettings):
    use_gpu: bool = False
    executable_source: str = "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/v0.0.1/llm-voice-chat"

class Train(BaseSettings):
    epochs: int = 1
    learning_rate: float = 1e-4


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
        args = [f"--{key} {value}" for key, value in self.model_dump().items()]
        return " ".join(args)


class Settings(BaseSettings):
    process: TrainingProcess
    model: Model 
    train: Train 
    data: Data

ENCODER = tiktoken.get_encoding("gpt2")

def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text.lower())

SENTIMENT_TO_INTEGER = {
    'pos': 1,
    'neg': 0
}

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
                sentiments.append(SENTIMENT_TO_INTEGER[sentiment])
                reviews.append(text)
            return sentiments, reviews
        yield fetch_data



@task
def download_rust_binary(url: str) -> str:
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    save_path = "train"
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    return save_path

@task
async def run_rust_binary(path_to_rust_binary: str, arguments: str):
    with subprocess.Popen(
        f'{path_to_rust_binary} {arguments}',
        shell=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    ) as process:
        while (code := process.poll()) is None:
            await asyncio.sleep(10)
        print(code)


@flow
async def training_loop(settings: Settings):
    path_to_rust_binary = download_rust_binary(settings.train.process.executable_source)
    await run_rust_binary(path_to_rust_binary, settings.model.to_cmd_args())

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
async def write_data(settings: Settings):
    logger = get_run_logger()
    logger.info("Partitioning dataset.")
    with dataset_partitioning(
        number_of_epochs=settings.train.epochs,
        number_of_partions=settings.train.data.preprocessing.slices,
        dataset_link = settings.train.data.source,
    ) as fetch_data:
        clean_safetensor_files()
        for epoch in range(settings.train.epochs):
            for slice in range(settings.train.data.preprocessing.slices):
                logger.info(f"Constructing slice {slice} for epoch {epoch}")
                data = fetch_and_preprocess_data(fetch_data, epoch, slice, settings.model.context_window)
                await wait_data_consumption()
                save_data(data)


@flow
async def sentiment_analysis(settings: Settings):
    parallel_subflows = [training_loop(settings), write_data(settings)]
    await asyncio.gather(*parallel_subflows)
    

def get_active_branch_name():

    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f: content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


import os
os.environ["LLMVC_ENVIRONMENT"] = "dev"

if __name__ == "__main__":

    if os.environ["LLMVC_ENVIRONMENT"] == "dev":
        sentiment_analysis.serve(name="sentiment-analysis-test")

    else:
        git_repo = GitRepository(
            url="https://github.com/Digital-Defiance/llm-voice-chat.git",
            branch = get_active_branch_name(),
        )

        sentiment_analysis_flow = sentiment_analysis.from_source(
            entrypoint="pipelines/sentiment_analysis/deploy.py:sentiment_analysis",
            source=git_repo,
        )

        sentiment_analysis_flow.deploy(
            name="sentiment-analysis",
            work_pool_name = "test",
        )
