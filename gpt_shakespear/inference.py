from train_config import ModelHandler
import torch
import torch.nn as nn
import mlflow.pytorch
import mlflow
import os
import tiktoken
import numpy as np
import torch.nn.functional as F

enc = tiktoken.get_encoding("gpt2")


logged_model = 'runs:/e951f26c18964b5ba58d7a562cac5b32/nanogpt_10'
nanoGPT = mlflow.pytorch.load_model(logged_model)
nanoGPT = nanoGPT.to('cpu')
nanoGPT.eval()

model_params = ModelHandler()

@torch.no_grad()
def generate(temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """

    sequence_s = "To be, or not to be, that is the question"
    sequence_s = enc.encode(sequence_s)
    sequence_1s = torch.tensor([sequence_s], dtype=torch.long, device='cpu')
    print(sequence_1s)

    max_new_tokens = 10
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        if sequence_1s.size(1) > model_params.words:
            sequence_1s = sequence_1s[:, -model_params.words:]

        # forward the model to get the logits for the index in the sequence
        logits_1st = nanoGPT(sequence_1s)
        logits_1ts = logits_1st.transpose(-1, -2)
        # pluck the logits at the final step and scale by desired temperature
        logits_1s = logits_1ts[:, -1, :] / temperature
        # optionally crop the logits to only the top k options

        # apply softmax to convert logits to (normalized) probabilities
        probs_1s = F.softmax(logits_1s, dim=-1)
        # sample from the distribution
        next_token_11 = torch.multinomial(probs_1s, num_samples=1)
        # append sampled index to the running sequence and continue
        sequence_1s = torch.cat((sequence_1s, next_token_11), dim=1)
    

    print(sequence_1s)
    return enc.decode(sequence_1s[0].tolist())

res = generate()
print(res)