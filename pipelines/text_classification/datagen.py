

from contextlib import contextmanager

import duckdb
from duckdb.typing import *
from numpy.random import default_rng

from prefect import get_run_logger, task, flow
import numpy as np
import tiktoken
from constants import SAVE_PATH
from inputs import Data, Train, Model
import uuid
from safetensors import torch as stt
from torch import Tensor
from typing import Literal
from functools import wraps, lru_cache
from torch.nn.utils.rnn import pad_sequence
from torch import tensor
from prefect.testing.utilities import prefect_test_harness

Sentiment = Literal["pos", "neg"]


ENCODER = tiktoken.get_encoding("gpt2")

def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text.lower())


def execute(func):

    @task(name=func.__name__)
    def prefect_task(conn: duckdb.DuckDBPyConnection, *args, **kwargs) -> list:
        logger = get_run_logger()
        sql_cmd = func(*args, **kwargs)
        logger.debug("Executing command: %", sql_cmd)
        return conn.execute(sql_cmd).fetchall()
    return prefect_task

def executemany(func):
    @wraps(func)
    @task
    def prefect_task(conn: duckdb.DuckDBPyConnection, *args,  **kwargs) -> list:
        logger = get_run_logger()
        sql_cmd, sql_args = func(*args, **kwargs)
        logger.debug("Executing command: %", sql_cmd)

        print(sql_cmd, sql_args)
        return conn.executemany(sql_cmd, sql_args)
    return prefect_task

@execute
def create_dataset_table(source: str):
    return f"""
        CREATE TABLE dataset AS
        SELECT id
        FROM '{source}';
    """

@execute
def add_permutation_column():
    return f"""
        ALTER TABLE dataset
        ADD COLUMN permutation
        INTEGER;
    """

@execute
def count():
    return f"""SELECT COUNT(*) FROM dataset"""


@executemany
def generate_permutation(rng, number_of_rows: int):
    permutation = rng.permutation(number_of_rows)
    return f"UPDATE dataset SET permutation = ? WHERE id = ?;", [
        (int(val) + 1, i + 1) for i, val in enumerate(permutation) 
    ]


@execute
def fetch_data(dataset_link: str, offset: int, limit: int) -> list[tuple[Sentiment, str]]:
    return f"""
        SELECT remote.sentiment, remote.review
        FROM '{dataset_link}' as remote
        INNER JOIN dataset ON dataset.id = remote.id
        ORDER BY permutation ASC
        OFFSET {offset} LIMIT {limit};
    """

# values: Iterator[tuple[int, int]]







@task
def parse_data_from_db(data_from_db):
    sentiments, reviews = [], []
    for sentiment, text in data_from_db:
        sentiment = 1 if sentiment == "pos" else 0
        sentiments.append(sentiment)
        text = text.strip()
        text = text.replace("<br />", "")
        text = text.lower()
        reviews.append(text)
    return sentiments, reviews


@task
def raw_data_to_tensor(raw_sentiments, raw_reviews):
    context_window: int = Model().context_window
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

@flow(flow_run_name="{name_prefix}prepare-{epochs}-epochs-{number_of_partions}-slices-{folder}")
def prepare_slices(conn, rng, epochs: int, number_of_partions: int, data_source: str, folder: str, name_prefix = ""):
    

    logger = get_run_logger()

    logger.info(f"Initializing table for {folder}")

    create_dataset_table(conn, data_source)

    add_permutation_column(conn)

    number_of_rows = count(conn)[0][0]


    limit = number_of_rows // number_of_partions
    step = 1

    logger.info(f"Generating data for {folder} ...")

    for epoch in range(epochs):
        for slice_idx in range(number_of_partions):

            logger.info(f"Constructing epoch {epoch} slice {slice_idx}")
            
            generate_permutation(conn, rng, number_of_rows)

            data_from_db: list[tuple[Sentiment, str]] = fetch_data(
                conn, 
                data_source, 
                offset=limit*slice_idx, 
                limit = number_of_rows // number_of_partions
            )

            sentiments, reviews = parse_data_from_db(data_from_db)
            safetensors = raw_data_to_tensor(sentiments, reviews)
            stt.save_file(safetensors, f"{folder}/{step}_output.safetensors")
            step += 1





import pytest

@pytest.mark.parametrize("epochs", [4])
@pytest.mark.parametrize("slices", [1])
def test_full(epochs: int, slices: int):
    import os
    import shutil

    SEED = 42

    raw_data = [
        (1, "pos", "A"),
        (2, "pos", "A"),
        (3, "neg", "B"),
        (4, "neg", "B"),
    ]

    folder = "tmp"
    if os.path.exists(folder):
        shutil.rmtree(folder)  

    os.mkdir(folder) 

    with duckdb.connect() as conn:

        conn.execute(f"""
            CREATE TABLE source
            (id INTEGER, sentiment VARCHAR, review VARCHAR);
        """)
        for row in raw_data:
            conn.execute("INSERT INTO source VALUES (?, ?, ?)", row)
        assert raw_data == conn.execute("SELECT * FROM source").fetchall(), "Mock source dataset failed sanity check."


        flow_rng = default_rng(seed=SEED)
        prepare_slices(conn, flow_rng, epochs, slices, "source", folder, name_prefix = "test-")



        dataset = conn.execute("SELECT id, permutation FROM dataset").fetchall()
        assert len(dataset) == len(raw_data), dataset

        # Verify that the permutation column has been properly constructed
        permutations, ids = [], []
        for idx, (dataset_id, permutation_idx) in enumerate(dataset):
            assert idx + 1 == dataset_id
            assert permutation_idx is not None, f"Permutation column was not constructed properly, value is None. {dataset=}"
            assert isinstance(permutation_idx, int)
            assert 1 <= permutation_idx <= 4
            permutations.append(permutation_idx)
            ids.append(dataset_id)
        ids.sort()
        permutations.sort()
        assert ids == permutations

        test_rng = default_rng(seed=SEED)

        for slice_idx in range(epochs*slices):
            safetensors_file = f"{folder}/{slice_idx+1}_output.safetensors"
            assert os.path.exists(safetensors_file), "Missing data from disk"
            data = stt.load_file(safetensors_file)
            
            for val_1, val_idx in zip(data['Y'], test_rng.permutation(len(raw_data))):
                val_2 = 1 if raw_data[val_idx][1] == "pos" else 0
                assert float(val_1) == val_2
        
            for token, sentiment in zip(data['X'], data['Y']):
                if token == 64:
                    assert sentiment == 1
                elif token == 65:
                    assert sentiment == 0














