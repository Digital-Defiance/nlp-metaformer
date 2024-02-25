import duckdb
from contextlib import contextmanager


DATASET_LINK = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet"


SENTIMENT_TO_INTEGER = {
    'pos': 1,
    'neg': 0
}


def set_seed(conn, seed):
    raise NotImplementedError

def create_table(conn):
    conn.sql(f"""
        CREATE TABLE dataset AS
        SELECT id
        FROM '{DATASET_LINK}';
    """)

def add_epoch_column(conn, epoch_idx, number_of_partitions):
    conn.sql(f"""
        ALTER TABLE dataset
        ADD COLUMN epoch_{epoch_idx}
        INTEGER DEFAULT trunc( {number_of_partions}*random());
    """)

def select_partition(conn, epoch_idx, slice_idx):
    return conn.sql(f"""
        SELECT sentiment, review
        FROM '{DATASET_LINK}' as remote
        JOIN dataset ON (dataset.id = remote.id)
        WHERE dataset.epoch_{epoch_idx}={slice_idx};
    """)
    

@contextmanager
def dataset_partitioning(number_of_epochs, number_of_partions, seed = 0.5):
    with duckdb.connect() as conn:
        set_seed(conn, seed)
        create_table(conn)

        # note: sampling is pre-computed here, data is fetched on demand with "fetch_data"
        for epoch_idx in range(number_of_epochs):
            add_epoch_column(conn, epoch_idx, number_of_partions)
        
        def fetch_data(epoch_idx: int, slice_idx: int):
            sentiments, reviews = [], []
            for sentiment, text in select_partition(conn, epoch_idx, slice_idx).fetchall():
                sentiments.append(SENTIMENT_TO_INTEGER[sentiment])
                reviews.append(text)
            return sentiments, reviews
        yield fetch_data


if __name__ == "__main__":
    with dataset_partitioning(number_of_epochs=2, number_of_partions=5) as fetch_data:
        sentiments, reviews = fetch_data(epoch_idx=0, slice_idx=3)