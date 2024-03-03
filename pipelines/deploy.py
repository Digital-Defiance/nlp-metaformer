import os
import asyncio
import subprocess
from pathlib import Path
import asyncio
from contextlib import contextmanager
from typing import Literal
import duckdb
from duckdb import DuckDBPyConnection
from duckdb.typing import *
from torch import Tensor, tensor, save
from torch.nn.utils.rnn import pad_sequence
from pydantic_settings.sources import YamlConfigSettingsSource
from pydantic_settings import BaseSettings
from prefect import flow, serve, get_run_logger, task, variables
from safetensors import torch as stt
import tiktoken
from prefect.runner.storage import GitRepository
from pathlib import Path

PathStr = str









class Optimizer(BaseSettings):
    learning_rate: float = 1e-4

class Preprocessing(BaseSettings):
    slices: int = 5
    batch_size: int = 32

class Data(BaseSettings):
    source: str = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet"
    output: str = "data.safetensors"
    preprocessing: Preprocessing

class Train(BaseSettings):
    epochs: int = 1
    optimizer: Optimizer
    data: Data

class Model(BaseSettings):
    attention_kind: Literal["Quadratic", "Metric", "ScaledDotProduct"] = "Quadratic"
    dimension: int = 32
    depth: int = 1
    heads: int = 2
    context_window: int = 300
    input_vocabolary: int = 60_000
    output_vocabolary: int = 5

class Run(BaseSettings):
    use_gpu: bool = False
    rust_binary: str = "/workspace/llm-voice-chat/pipelines/llm-voice-chat"
    config_file: str = "/workspace/llm-voice-chat/config.yml"

class Settings(BaseSettings):
    run: Run
    use_gpu: bool = False
    train: Train
    model: Model

    @classmethod
    def load(cls, path = "config.yml"):    
        data = YamlConfigSettingsSource(BaseSettings, "config.yml")()
        return cls(**data)



ENCODER = tiktoken.get_encoding("gpt2")

def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text.lower())

SENTIMENT_TO_INTEGER = {
    'pos': 1,
    'neg': 0
}

def set_seed(conn: DuckDBPyConnection, seed: float) -> None:
    conn.execute(f"SELECT setseed(?) as ign;", [seed])

def create_table(conn: DuckDBPyConnection, dataset_link: str):
    conn.execute(f"""
        CREATE TABLE dataset AS
        SELECT id
        FROM '{dataset_link}';
    """)

def add_epoch_column(conn: DuckDBPyConnection, epoch_idx: int, number_of_partions: int):
    conn.execute(f"""
        ALTER TABLE dataset
        ADD COLUMN epoch_{epoch_idx}
        INTEGER DEFAULT trunc( {number_of_partions}*random() );
    """)

def select_partition(conn: DuckDBPyConnection, dataset_link: str, epoch_idx: int, slice_idx: int):
    return conn.execute(f"""
        SELECT sentiment, review
        FROM '{dataset_link}' as remote
        JOIN dataset ON (dataset.id = remote.id)
        WHERE dataset.epoch_{epoch_idx}={slice_idx};
    """)

@contextmanager
def dataset_partitioning(number_of_epochs, number_of_partions, dataset_link: str, seed = 0.5):
    with duckdb.connect() as conn:
        conn: DuckDBPyConnection
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

@flow
async def training_loop(settings: Settings):
    logger = get_run_logger()
    logger.info("Training loop flow started.")

    with subprocess.Popen(
        f'{settings.run.rust_binary} --path {settings.run.config_file}',
        shell=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    ) as process:
        while (code := process.poll()) is None:
            await asyncio.sleep(10)

@task
def clean_safetensor_files(path_to_data: PathStr):
    logger = get_run_logger()

    if Path(path_to_data).exists():
        logger.info("Removing: " + path_to_data)
        os.remove(file)

@task
async def wait_data_consumption():
    logger = get_run_logger()
    logger.info("Waiting for slice.safetensors to be consumed")
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
    logger = get_run_logger()
    stt.save_file(data, SAVE_PATH)
    logger.info("Saved new slice.safetensors")


@flow
async def write_data(settings: Settings):
    logger = get_run_logger()
    logger.info("Started flow.")
    logger.info("Partitioning dataset.")
    with dataset_partitioning(
        number_of_epochs=settings.train.epochs,
        number_of_partions=settings.train.data.preprocessing.slices,
        dataset_link = settings.train.data.source,
    ) as fetch_data:
        clean_safetensor_files()
        for epoch in range(number_of_epochs):
            for slice in range(number_of_partions):
                logger.info(f"Constructing slice {slice} for epoch {epoch}")
                data = fetch_and_preprocess_data(
                    fetch_data,
                    epoch,
                    slice,
                    settings.model.context_window
                )
                await wait_data_consumption()
                save_data(data)


@flow
async def sentiment_analysis(settings: Settings):
    logger = get_run_logger()
    logger.info(settings)
    parallel_subflows = [training_loop(settings), write_data(settings)]
    await asyncio.gather(*parallel_subflows)
    
    

def get_active_branch_name():

    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f: content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]

if __name__ == "__main__":

    git_repo = GitRepository(
        url="https://github.com/Digital-Defiance/llm-voice-chat.git",
        branch = get_active_branch_name(),
    )

    sentiment_analysis_flow = sentiment_analysis.from_source(
        entrypoint="pipelines/deploy.py:sentiment_analysis",
        source=git_repo,
    )

    sentiment_analysis_flow.deploy(
        name="sentiment-analysis",
        work_pool_name = "test",
    )
