import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
from dotenv import load_dotenv
from tqdm import tqdm
from model import NanoGPT
from system_parameters import DEVICE
from train_config import TrainConfiguration, ModelHandler, MLFlowHandler
import torch

load_dotenv()
torch.autograd.set_detect_anomaly(True)

def generate_data(params, batches = 32):
  for _ in range(batches):
    sequence = torch.randint(0, params.tokens, (batches, params.words,)).to(DEVICE)
    sorted_matrix, _ = torch.sort(sequence, dim=1)  # Sort along columns
    yield sequence.to(DEVICE), sorted_matrix.to(DEVICE)


with MLFlowHandler.continue_run() as mlflow_handler:
    model_params = ModelHandler.load_from_mlflow(mlflow_handler)
    train_params = TrainConfiguration.load_from_mlflow(mlflow_handler)
    nanoGPT = NanoGPT(model_params).to(DEVICE)
    run = mlflow_handler.get_run()
    last_epoch = run['step'].max()
    if last_epoch:
        model_uri = f"runs:/{mlflow_handler._run_id}/gpt_array_sorter_epoch_{last_epoch}"
        state_dict = mlflow.pytorch.load_model(model_uri)
        nanoGPT.load_state_dict(state_dict)
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0

    optimizer = torch.optim.Adam(nanoGPT.parameters(), lr=train_params.learning_rate)
    if train_params.loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function {train_params.loss_function}")

    for epoch in range(start_epoch, train_params.number_of_epochs):
        data_generator = generate_data(batches=train_params.number_of_batches)
        progress_bar = tqdm(data_generator, desc=f"Epoch {epoch}", leave=True)
        for batch, targets in progress_bar:
            optimizer.zero_grad()
            outputs = nanoGPT(batch)
            loss = loss_function(outputs.transpose(-1, -2), targets)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
    
        mlflow.pytorch.log_model(nanoGPT, f"gpt_array_sorter_epoch_{epoch}")
        mlflow.log_metric("loss", loss.item(), epoch)

