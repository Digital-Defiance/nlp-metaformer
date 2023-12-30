import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
from data import generate_data
from dotenv import load_dotenv
from tqdm import tqdm
from mlflow_handler import MLFlowHandler
from model_handler import ModelHandler

load_dotenv()

torch.autograd.set_detect_anomaly(True)

with MLFlowHandler.continue_run() as mlflow_handler:
    model_parameters = mlflow_handler.get_parameter("model_parameters")
    nanoGPT = ModelHandler.create_from_parameters(model_parameters)
    epochs = mlflow_handler.get_parameter("epochs")
    NUMBER_OF_BATCHES = mlflow_handler.get_parameter("batches")
    loss_function = mlflow_handler.get_parameter("loss_function")
    LEARNING_RATE = mlflow_handler.get_parameter("learning_rate")
    run = mlflow_handler.get_run()
    last_epoch = run['step'].max()
    if last_epoch:
        model_uri = f"runs:/{mlflow_handler._run_id}/gpt_array_sorter_epoch_{last_epoch}"
        state_dict = mlflow.pytorch.load_model(model_uri)
        nanoGPT.load_state_dict(state_dict)
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0

    optimizer = torch.optim.Adam(nanoGPT.parameters(), lr=LEARNING_RATE)
    if loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function {loss_function}")

    for epoch in range(start_epoch, epochs):
        data_generator = generate_data(batches=NUMBER_OF_BATCHES)
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

