import duckdb
from contextlib import contextmanager
from duckdb import DuckDBPyConnection

from torch import Tensor, tensor
import unittest



DATASET_LINK = "https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet"


SENTIMENT_TO_INTEGER = {
    'pos': 1,
    'neg': 0
}


def set_seed(conn: DuckDBPyConnection, seed: float) -> None:
    conn.execute(f"SELECT setseed(?) as ign;", [seed])

def create_table(conn: DuckDBPyConnection):
    conn.execute(f"""
        CREATE TABLE dataset AS
        SELECT id
        FROM '{DATASET_LINK}';
    """)

def add_epoch_column(conn: DuckDBPyConnection, epoch_idx: int, number_of_partions: int):
    conn.execute(f"""
        ALTER TABLE dataset
        ADD COLUMN epoch_{epoch_idx}
        INTEGER DEFAULT trunc( {number_of_partions}*random() );
    """)

def select_partition(conn: DuckDBPyConnection, epoch_idx: int, slice_idx: int):
    return conn.execute(f"""
        SELECT sentiment, review
        FROM '{DATASET_LINK}' as remote
        JOIN dataset ON (dataset.id = remote.id)
        WHERE dataset.epoch_{epoch_idx}={slice_idx};
    """)
    

@contextmanager
def dataset_partitioning(number_of_epochs, number_of_partions, seed = 0.5):
    
    with duckdb.connect() as conn:
        conn: DuckDBPyConnection
        set_seed(conn, seed)
        create_table(conn)

        # note: sampling is pre-computed here, data is fetched on demand with "fetch_data"
        for epoch_idx in range(number_of_epochs):
            add_epoch_column(conn, epoch_idx, number_of_partions)
        
        def fetch_data(epoch_idx: int, slice_idx: int) -> tuple[list[int], list[str]]:
            sentiments, reviews = [], []
            for sentiment, text in select_partition(conn, epoch_idx, slice_idx).fetchall():
                sentiments.append(SENTIMENT_TO_INTEGER[sentiment])
                reviews.append(text)
            return sentiments, reviews
        yield fetch_data
        
        

        
class TestCase(unittest.TestCase):
    
    
    def test_setseed(self):
        for seed in [0.1, -0.4]:
            with duckdb.connect() as conn:
                set_seed(conn, seed)
                result_1 = conn.execute("SELECT random() as random_number FROM range(5);").fetchnumpy()
                result_1 = result_1["random_number"]
                
            with duckdb.connect() as conn:
                set_seed(conn, seed)
                result_2 = conn.execute("SELECT random() as random_number FROM range(5);").fetchnumpy()
                result_2 = result_2["random_number"]
            
            
            for x, y in zip(result_1, result_2):
                assert x == y
        
    
    def test_partition_run(self):
        with dataset_partitioning(number_of_epochs=1, number_of_partions=5) as fetch_data:
            s2, r2 = fetch_data(epoch_idx=0, slice_idx=3)
            
            
        with dataset_partitioning(number_of_epochs=1, number_of_partions=5) as fetch_data:
            s1, r1 = fetch_data(epoch_idx=0, slice_idx=3)

        assert r1[0] == r2[0]
        assert s1[0] == s2[0]
            
            
        assert isinstance(r1[0], str), r1[0]
        assert isinstance(s2[0], int), s2[0]


if __name__ == "__main__":
    unittest.main()
    
    
