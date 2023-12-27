import torch

from system_parameters import DEVICE

def generate_data(params, batches = 32):
  for _ in range(batches):
    sequence = torch.randint(0, params.tokens, (batches, params.words,)).to(DEVICE)
    sorted_matrix, _ = torch.sort(sequence, dim=1)  # Sort along columns
    yield sequence.to(DEVICE), sorted_matrix.to(DEVICE)