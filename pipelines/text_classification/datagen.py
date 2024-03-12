

from contextlib import contextmanager

import duckdb
from duckdb.typing import *
from prefect import get_run_logger, task
import numpy as np
import tiktoken
from constants import SAVE_PATH
from env import Data, Train, Model
import uuid

ENCODER = tiktoken.get_encoding("gpt2")

def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text.lower())

class DatasetConnection:

    

    def __init__(self, conn: duckdb.DuckDBPyConnection, dataset_link: str):
        logger = get_run_logger()

        self.generated_epochs = set()

        self.dataset_link = dataset_link
        self.conn = conn
        self.uuid = "".join([x for x in uuid.uuid4().hex if not x.isnumeric()])

        logger.debug(self.uuid)
        logger.debug(self.dataset_link)


        cmd = f"""
            CREATE TABLE {self.uuid} AS
            SELECT id
            FROM '{self.dataset_link}';
        """
        logger.debug(cmd)
        self.conn.execute(cmd)


        cmd = f"""SELECT COUNT(*) FROM {self.uuid}"""
        logger.debug(cmd)
        self.count = self.conn.execute(cmd).fetchall()[0][0]


    def generate_randomization(self, total_epochs: int, seed: int = 42):

        np.random.seed(seed)

    def select_partition(self, epoch_idx: int, start: int, limit: int):

        if epoch_idx not in self.generated_epochs:
            
            self.conn.execute(f"""
                ALTER TABLE {self.uuid}
                ADD COLUMN epoch_{epoch_idx}
                INTEGER;
            """)

            permutation = np.random.permutation(self.count)
            print(epoch_idx, permutation)
            values_to_insert = [(int(val) + 1, i + 1) for i, val in enumerate(permutation)]
            sql_command = f"UPDATE {self.uuid} SET epoch_{epoch_idx} = ? WHERE id = ?;"
            self.conn.executemany(sql_command, values_to_insert)
            self.generated_epochs.add(epoch_idx)

        return self.conn.execute(f"""
            SELECT remote.sentiment, remote.review
            FROM '{self.dataset_link}' as remote
            INNER JOIN {self.uuid} ON {self.uuid}.id = remote.id
            ORDER BY epoch_{epoch_idx} ASC
            OFFSET {start} LIMIT {limit};
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
                sentiment = 1 if sentiment == "pos" else 0
                sentiments.append(sentiment)
                text = text.strip()
                text = text.replace("<br />", "")
                text = text.lower()
                reviews.append(text)
            return sentiments, reviews
        yield fetch_data


@task
def fetch_and_preprocess_data(fetch_data: callable, epoch: int, slice: int):
    from torch.nn.utils.rnn import pad_sequence
    from torch import tensor

    context_window: int = Model().context_window


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
    stt.save_file(data, f"tmp/{idx}_" + SAVE_PATH)



@task
def write_training_slices():

    data = Data()
    train = Train()


    def yield_context():
        step = 1
        for epoch in range(train.epochs):
            for slice_idx in range(data.slices):
                yield step, epoch, slice_idx
                step += 1


    with dataset_partitioning(
        number_of_epochs = train.epochs,
        number_of_partions = data.slices,
        dataset_link = data.train_source
    ) as fetch_data:
        logger = get_run_logger()
        for step, epoch, slice in yield_context():
            logger.info(f"Constructing slice {slice} for epoch {epoch}")
            processed_data = fetch_and_preprocess_data.fn(fetch_data, epoch, slice)
            save_data.fn(step, processed_data)





@task
def write_test_slices():

    data = Data()
    train = Train()

    with dataset_partitioning(
        number_of_epochs = train.epochs,
        number_of_partions = data.slices,
        dataset_link = data.test_source
    ) as fetch_data:
        for slice_idx in range(data.slices):
            processed_data = fetch_and_preprocess_data.fn(
                fetch_data = fetch_data, 
                epoch = 0,
                slice = slice_idx,
            )
            save_data.fn(
                idx = -(slice_idx + 1),
                data = processed_data
            )



@task
def prepare_validation_slice():

    data = Data()

    with dataset_partitioning(
        number_of_epochs = Train().epochs,
        number_of_partions = data.slices,
        dataset_link = data.test_source
    ) as fetch_data:

        data = fetch_and_preprocess_data.fn(
            fetch_data=fetch_data,
            epoch=0,
            slice=0,
        )

        save_data.fn(
            idx = 0,
            data = data,
        )

