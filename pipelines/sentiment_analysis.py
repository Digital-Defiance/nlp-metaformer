from prefect import flow, task
import duckdb

import duckdb
from duckdb.typing import *
import tiktoken

import asyncio

encoder = tiktoken.get_encoding("gpt2")



def encode_text(text: str) -> list[int]:
    text = text.lower()
    return encoder.encode(text)




def imbd():
    number_of_partions = 5
    number_of_epochs = 3
    
    conn.sql("""
        CREATE TABLE dataset AS (
            SELECT id FROM 'https://github.com/Digital-Defiance/IMBd-dataset/raw/main/dataset/dataset.parquet'
        );
    """)

    for i in range(number_of_epochs):
        conn.sql(f"""
            ALTER TABLE dataset
            ADD COLUMN epoch_{i} 
            INTEGER DEFAULT trunc( {number_of_partions}*random() );
        """)


@flow
async def training_loop():
    import os
    import subprocess
    command = 'while true; do echo "Hello, world!"; sleep 2; done'
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
        while (code := process.poll()) is None:
            print("Process is running.")
            await asyncio.sleep(30)


@flow
async def main():
    
    dataset = "imbd"
    
    if dataset == "imbd":
        preprocessing_loop = imbd
    
    parallel_subflows = [training_loop(), preprocessing_loop()]
    await asyncio.gather(*parallel_subflows)

if __name__ == "__main__":
    main_flow_state = asyncio.run(main())