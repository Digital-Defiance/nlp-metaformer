import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
from tqdm import tqdm
from model import NanoGPT
import torch
import logging
from contextlib import contextmanager
from train_config import TrainConfiguration, ModelHandler, MLFlowSettings
from mlflow.entities import RunStatus
import requests
import torch
from pydantic_settings import BaseSettings
from typing import Literal, Iterator
import tiktoken
import numpy as np



# ----------------- SETTINGS ----------------- #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")



# ----------------- ENVIRONMENT ----------------- #

class EnvironmentSettings(BaseSettings):
    environment: Literal["local", "aws"] = "aws"
    create_run: bool = False

    class Config:
        env_prefix = "MACHINE_"


environment_settings = EnvironmentSettings()
mlflow_settings = MLFlowSettings()

if not mlflow_settings.run_id and not environment_settings.create_run:
    raise ValueError("MLFLOW_RUN_ID environment variable is not set")


if environment_settings.environment == "aws":
    # Get the token for the metadata service (AWS)
    url = "http://169.254.169.254/latest/api/token"
    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
    token_response = requests.put(url, headers=headers)
    token_response.raise_for_status()
    token = token_response.text


# ----------------- EVENT HANDLING ----------------- #


class MyException(Exception):
    """ Base class for all exceptions in this module. """

class PauseTraining(MyException):
    """ Raised when the training should be paused. """

@contextmanager
def exception_controlled_run() -> Iterator[mlflow.ActiveRun]:
    """ This controls what the worker comunicates back to main."""

    run_kwargs = {
        "run_id": mlflow_settings.run_id,
        "experiment_id": mlflow_settings.experiment_id,
        "log_system_metrics": mlflow_settings.log_system_metrics,
    }

    def set_status(status: RunStatus):
        status = RunStatus.to_string(status)
        mlflow.tracking.MlflowClient().set_terminated(mlflow_settings.run_id, status=status)

    try:
        with mlflow.start_run(**run_kwargs) as run:
            yield run
    except PauseTraining:
        set_status(RunStatus.SCHEDULED)
        raise SystemExit
    except MyException as e:
        set_status(RunStatus.FAILED)
        raise e
    except Exception as e:
        set_status(RunStatus.FAILED)
        raise e

# ----------------- DATA ----------------- #

enc = tiktoken.get_encoding("gpt2")
input_file_path = "raw_data.txt"

with open(input_file_path, 'r') as file:
    text: str = file.read()

size_of_text = len(text)
thresh_size = int(size_of_text * 0.9)
train_data, val_data = text[:thresh_size], text[thresh_size:]


train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
train_ids = np.array(train_ids, dtype=np.int32)
val_ids = np.array(val_ids, dtype=np.int32)

train_ids = torch.from_numpy(train_ids).to(DEVICE)
val_ids = torch.from_numpy(val_ids).to(DEVICE)


def generate_batch(data_s: torch.Tensor):
    batch_size = train_params.number_of_batches
    shape = (batch_size,)
    start_indices_b = torch.randint(0, len(data_s) - model_params.words, shape)
    end_indices_b = start_indices_b + model_params.words
    shape = (batch_size, model_params.words)
    in_sequence_bw = torch.zeros(shape, dtype=torch.int64, device=DEVICE)
    out_sequence_bw = torch.zeros(shape, dtype=torch.int64, device=DEVICE)

    for i in range(train_params.number_of_batches):
        start, end = start_indices_b[i], end_indices_b[i]
        in_sequence_bw[i] = data_s[start:end]
        out_sequence_bw[i] = data_s[start + 1:end + 1]

    return in_sequence_bw, out_sequence_bw

def generate_epoch():
    for _ in range(train_params.number_of_epochs):
        yield generate_batch(train_ids)

# ----------------- TRAINING ----------------- #

with exception_controlled_run() as run:

    # ----------------- INIT ----------------- #

    if environment_settings.create_run:
        mlflow_settings.run_id = run.info.run_id
        TrainConfiguration().save_to_mlflow()
        ModelHandler().save_to_mlflow()

    train_params = TrainConfiguration.load_from_mlflow()
    model_params = ModelHandler.load_from_mlflow()
    model_params.tokens = enc.max_token_value

    if train_params.loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function {train_params.loss_function}")

    # Get the last epoch from the previous run
    last_epoch: float | None = mlflow.get_run(run.info.run_id).data.metrics.get('epoch', None)
    if last_epoch is not None:
        last_epoch = int(last_epoch)
        logger.debug("Last epoch is %s", last_epoch)
        model_uri = f"runs:/{mlflow_settings.run_id}/nanogpt_{last_epoch}"
        nanoGPT: NanoGPT = mlflow.pytorch.load_model(model_uri)
        nanoGPT: NanoGPT = nanoGPT.to(DEVICE)
        start_epoch = last_epoch + 1
        logger.info(f"Loaded model from {model_uri}")
    else:
        nanoGPT = NanoGPT(model_params).to(DEVICE)
        start_epoch = 0
    
    optimizer = torch.optim.Adam(nanoGPT.parameters(), lr=train_params.learning_rate)

    @contextmanager
    def zero_grad(optimizer: torch.optim.Optimizer) -> Iterator[None]:
        """ Ensures that I don't forget to zero the gradients :)"""

        optimizer.zero_grad()
        yield
        optimizer.step()

    # ----------------- TRAINING LOOP ----------------- #

    for epoch in range(start_epoch, train_params.number_of_epochs):
        data_gen = generate_epoch()
        data_gen_progressbar = tqdm(data_gen, desc=f"Epoch {epoch}", leave=True)
        for in_sequence_bw, out_sequence_bw in data_gen_progressbar:
            
            with zero_grad(optimizer):
                pred_logits_bwt = nanoGPT(in_sequence_bw)
                # cross entropy expects (batch, classes, sequence)
                pred_logits_btw = pred_logits_bwt.transpose(-1, -2) 
                loss = loss_function(pred_logits_btw, out_sequence_bw)
                loss.backward()
    
            data_gen_progressbar.set_postfix(loss=loss.item())

            if environment_settings.environment == "local":
                continue

            # Check if the instance is scheduled to be terminated
            try:
                url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
                headers = {"X-aws-ec2-metadata-token": token}
                requests.get(url, headers=headers).raise_for_status()
            except requests.exceptions.HTTPError as err:
                if not err.response.status_code == 404:
                    raise PauseTraining                    
        
        # Save the model and log the loss
        mlflow.pytorch.log_model(nanoGPT, f"nanogpt_{epoch}")
        mlflow.log_metric("loss", loss.item(), epoch)
        mlflow.log_metric("epoch", epoch, epoch)
 
    


