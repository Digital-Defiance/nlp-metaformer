from prefect import flow, task
import duckdb

import duckdb
from duckdb.typing import *
import tiktoken
import os
import asyncio
import os
import subprocess
from pathlib import Path
from safetensors import torch as stt
from prefect import flow, serve, get_run_logger
from sentiment_analysis.imbd import dataset_partitioning
from torch import tensor, save
from torch.nn.utils.rnn import pad_sequence

encoder = tiktoken.get_encoding("gpt2")

RUST_BINARY: str = "/workspace/llm-voice-chat/pipelines/llm-voice-chat"
SAVE_PATH = f"./pipelines/slice.safetensors"



def encode_text(text: str) -> list[int]:
    text = text.lower()
    return encoder.encode(text)

@task
def start_rust_binary_subprocess():
    return subprocess.Popen(
        f'{RUST_BINARY} --path /workspace/llm-voice-chat/config.yml',
        shell=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )


@task
def poll_process_for_logs(process):
    logger = get_run_logger()
    logger.info("Logs")
    
@task
async def defer_control_of_event_loop():
    logger = get_run_logger()
    logger.info("stuff")
    await asyncio.sleep(10)

@flow
async def training_loop():
    logger = get_run_logger()
    logger.info("start")

    process = start_rust_binary_subprocess()
    while (code := process.poll()) is None:
        # poll_process_for_logs(process)
        await defer_control_of_event_loop()
            
@task
def clean_safetensor_files():
    logger = get_run_logger()
    for file in os.listdir("."):
        if file.endswith(".safetensors"):
            logger.info("Removing " + file)
            os.remove(file)


@task
async def wait_data_consumption():
    logger = get_run_logger()
    logger.info("Waiting for slice.safetensors to be consumed")
    while Path(SAVE_PATH).exists():
        await asyncio.sleep(1)
        
@task
def generate_data(fetch_data, epoch, slice):
    sentiments, reviews = fetch_data(epoch, slice)
    sentiments = tensor(sentiments)
    
    tmp = []
    for text in reviews:
        text = encode_text(text)[0:300]
        text = tensor(text)
        tmp.append(text)

    reviews = tmp

    reviews = pad_sequence(reviews, batch_first=True)                
    return {
        'Y': sentiments,
        "X": reviews
    }

@task
def save_data(data):
    logger = get_run_logger()
    stt.save_file(data, SAVE_PATH)
    logger.info("Saved new slice.safetensors")

@flow
async def write_imbd_data():
    logger = get_run_logger()
    logger.info("Started flow.")

    number_of_epochs = 5
    number_of_partions = 15
    
    logger.info("Partitioning dataset.")
    with dataset_partitioning(number_of_epochs=number_of_epochs, number_of_partions=number_of_partions) as fetch_data:
        clean_safetensor_files()
        for epoch in range(number_of_epochs):
            for slice in range(number_of_partions):
                logger.info(f"Constructing slice {slice} for epoch {epoch}")
                data = generate_data(fetch_data, epoch, slice)
                await wait_data_consumption()
                save_data(data)


                

@flow
async def main():
    
    dataset = "imbd"
    
    if dataset == "imbd":
        preprocessing_loop = write_imbd_data

    parallel_subflows = [training_loop(), preprocessing_loop()]
    await asyncio.gather(*parallel_subflows)

if __name__ == "__main__":
    # main_flow_state = asyncio.run(main())
    main_deployment = main.to_deployment(name="main")
    serve(main_deployment)