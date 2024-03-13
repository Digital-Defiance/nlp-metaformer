
import duckdb
from duckdb.typing import *
from numpy.random import default_rng
from prefect import get_run_logger, task, flow
import tiktoken
from inputs import  Model
from safetensors import torch as stt
from typing import Literal
from functools import wraps
from torch.nn.utils.rnn import pad_sequence
from torch import tensor
import pytest


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
    excess = number_of_rows % number_of_partions




    step = 1

    logger.info(f"Generating data for {folder} ...")

    for epoch in range(epochs):
        
        generate_permutation(conn, rng, number_of_rows)
        
        for slice_idx in range(number_of_partions):

            logger.info(f"Constructing epoch {epoch} slice {slice_idx}")
            

            data_from_db: list[tuple[Sentiment, str]] = fetch_data(
                conn, 
                data_source, 
                offset = limit*slice_idx, 
                limit = limit if not slice_idx == number_of_partions - 1 else limit + excess
            )

            sentiments, reviews = parse_data_from_db(data_from_db)
            safetensors = raw_data_to_tensor(sentiments, reviews)
            stt.save_file(safetensors, f"{folder}/{step}_output.safetensors")
            step += 1

            




@pytest.mark.parametrize("epochs", [2, 1])
@pytest.mark.parametrize("slices", [3, 2, 1])
def test_full(epochs: int, slices: int):
    import os
    import shutil

    SEED = 42

    POS_SENTIMENT = 1
    NEG_SENTIMENT = 0
    A_TOKEN = 64
    B_TOKEN = 65

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
            assert isinstance(permutation_idx, int), "Invalid type for permutation index."
            assert 1 <= permutation_idx <= 4, f"Found invalid permutation index: {permutation_idx}"
            permutations.append(permutation_idx)
            ids.append(dataset_id)
    
        ids.sort()
        permutations.sort()
        assert ids == permutations


        # replicate the permutation column 
        test_rng = default_rng(seed=SEED)

        slice_idx= 1
        for _ in range(epochs):
            
            # to reconstruct the data from the slices
            epoch_data = []

            for _ in range(slices):
                safetensors_file = f"{folder}/{slice_idx}_output.safetensors"
                assert os.path.exists(safetensors_file), f"Missing data from disk: {safetensors_file}"

                data = stt.load_file(safetensors_file)

                assert 'X' in data
                assert 'Y' in data

    
                for val in data['Y']:
                    epoch_data.append(float(val))

                for token, sentiment in zip(data['X'], data['Y']):
                    if token == A_TOKEN:
                        assert sentiment == POS_SENTIMENT
                    elif token == B_TOKEN:
                        assert sentiment == NEG_SENTIMENT
                    else:
                        assert False, f"Invalid token: {token}"
                
                slice_idx += 1
        
        
            assert len(epoch_data) == len(raw_data), "Generated data does not have the correct lenght"
            assert sum(epoch_data) == 2., "Generated data has incorrect values"


            epoch_permutation = test_rng.permutation(len(raw_data))
        
            for val_1, val_idx in zip(epoch_data, epoch_permutation):
                val_2 = POS_SENTIMENT if raw_data[val_idx][1] == "pos" else NEG_SENTIMENT
                assert float(val_1) == float(val_2), epoch_data

                



