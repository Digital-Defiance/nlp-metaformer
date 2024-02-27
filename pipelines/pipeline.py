from prefect import flow, task
import duckdb

import duckdb
from duckdb.typing import *
import tiktoken
import os
import asyncio

from safetensors import torch as stt

encoder = tiktoken.get_encoding("gpt2")



def encode_text(text: str) -> list[int]:
    text = text.lower()
    return encoder.encode(text)





@flow
async def training_loop():
    import os
    import subprocess
    command = 'while true; do echo "Hello, world!"; sleep 2; done'
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
        while (code := process.poll()) is None:
            print("Process is running.")
            await asyncio.sleep(10)



@flow
async def write_imbd_data():
    from sentiment_analysis.imbd import dataset_partitioning
    from torch import tensor, save
    from torch.nn.utils.rnn import pad_sequence
    
    number_of_epochs = 5
    number_of_partions = 15
    
    with dataset_partitioning(number_of_epochs=number_of_epochs, number_of_partions=number_of_partions) as fetch_data:
        
        
        for file in os.listdir("."):
            if file.endswith(".safetensors"):
                print("Removing ", file)
                os.remove(file)
        
        for epoch in range(number_of_epochs):
            for slice in range(number_of_partions):
                sentiments, reviews = fetch_data(epoch, slice)
                sentiments = tensor(sentiments)
                
                tmp = []
                for text in reviews:
                    text = encode_text(text)[0:300]
          
                    text = tensor(text)
                    tmp.append(text)
                reviews = tmp
                # reviews = tensor(reviews)
                name = f"epoch_{epoch}_slice_{slice}.safetensors"
                
                reviews = pad_sequence(reviews, batch_first=True)
                print(reviews.size())
                
                state_dict = {
                    'Y': sentiments,
                    "X": reviews
                }
                stt.save_file(state_dict, name)
                
                
                await asyncio.sleep(0)

                while len(
                    list(x for x in os.listdir(".") if x.endswith(".safetensors"))
                ) > 5:
                    await asyncio.sleep(1)

                
                
                

@flow
async def main():
    
    dataset = "imbd"
    
    if dataset == "imbd":
        preprocessing_loop = write_imbd_data

    parallel_subflows = [training_loop(), preprocessing_loop()]
    await asyncio.gather(*parallel_subflows)

if __name__ == "__main__":
    main_flow_state = asyncio.run(main())