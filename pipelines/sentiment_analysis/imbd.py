import duckdb
from contextlib import contextmanager


@contextmanager
def get_connection(number_of_epochs, number_of_partions):
    with duckdb.connect() as conn:

        conn.sql("""
            CREATE TABLE dataset AS
            SELECT id
            FROM 'https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet';
        """)

        for i in range(number_of_epochs):
            conn.sql(f"""
                ALTER TABLE dataset
                ADD COLUMN epoch_{i}
                INTEGER DEFAULT trunc( {number_of_partions}*random());
            """)

        to_integer = {
            'pos': 1,
            'neg': 0
        }
        
        def yield_partition(epoch: int, idx: int):
            for sentiment, text in conn.sql(f"""
                SELECT sentiment, review
                FROM 'https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet' as remote
                JOIN dataset ON (dataset.id = remote.id)
                WHERE dataset.epoch_{epoch}={idx};
            """).fetchall():
                yield to_integer[sentiment], text
        yield yield_partition


if __name__ == "__main__":
    with get_connection(2, 5) as yield_partition:
        for sentiment, text in yield_partition(0, 3):
            print(sentiment, text)
            break

