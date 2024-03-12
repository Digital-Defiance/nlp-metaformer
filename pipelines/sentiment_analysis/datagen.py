
from training_loop import run_rust_binary, make_rust_executable, download_rust_binary

from contextlib import contextmanager
from typing import Literal

import duckdb
from duckdb.typing import *
import mlflow
from pydantic_settings import BaseSettings
from prefect import flow, get_run_logger, task
import numpy as np
import tiktoken


from constants import SAVE_PATH, DEV_RUST_BINARY
from env import Data, Train, Settings, Model, TrainingProcess, MLFLowSettings


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

