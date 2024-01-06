from train_config import ModelHandler
import torch
import mlflow.pytorch
import mlflow
import tiktoken
import torch.nn.functional as F
from tqdm import tqdm

gpt2_encoder = tiktoken.get_encoding("gpt2")
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
logged_model = 'runs:/0dd2b64253584cd39c52fc1482c035ae/nanogpt_180'
nanoGPT = mlflow.pytorch.load_model(logged_model, map_location=DEVICE)
nanoGPT = nanoGPT.to('cpu')
nanoGPT.eval()

model_params = ModelHandler(words = 1024)

@torch.no_grad()
def generate(sequence_s = "The meaning of life is", max_new_tokens = 400):
    sequence_s = gpt2_encoder.encode(sequence_s)
    sequence_1s = torch.tensor([sequence_s], dtype=torch.long, device=DEVICE)
    so_far = sequence_1s[0].tolist()

    for _ in tqdm(range(max_new_tokens), leave=True):

        if sequence_1s.size(1) > model_params.words:
            sequence_1s = sequence_1s[:, -model_params.words:]

        logits_1st = nanoGPT(sequence_1s)
        logits_1t = logits_1st[:, -1, :]
        probs_1s = F.softmax(logits_1t, dim=-1)
        next_token_11 = torch.multinomial(probs_1s, num_samples=1)
        sequence_1s = torch.cat((sequence_1s, next_token_11), dim=1)
        so_far.append(next_token_11.item())
    
    return gpt2_encoder.decode(so_far, errors="replace")

res = generate()
print(res)
