import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
from tqdm import tqdm
import torch
from contextlib import contextmanager
from train.config import TrainConfiguration, MLFlowSettings
from model import ModelFactory, gpt2_encoder
from mlflow.entities import RunStatus
import requests
import torch
from typing import Iterator
import tiktoken
import numpy as np
from logger import get_logger



# ----------------- SETTINGS ----------------- #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")


mlflow_settings = MLFlowSettings()
model_factory = ModelFactory()
train_configuration = TrainConfiguration()

if not mlflow_settings.is_local:
    # Get the token for the metadata service (AWS)
    url = "http://169.254.169.254/latest/api/token"
    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
    token_response = requests.put(url, headers=headers)
    token_response.raise_for_status()


    url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    headers = {"X-aws-ec2-metadata-token": token_response.text}

    def instance_will_terminate():
        try:
            requests.get(url, headers=headers).raise_for_status()
            return False
        except requests.exceptions.HTTPError as err:
            if not err.response.status_code == 404:
                return True


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

gpt2_encoder = tiktoken.get_encoding("gpt2")
input_file_path = "raw_data.txt"

with open(input_file_path, 'r') as file:
    text: str = file.read()

size_of_text = len(text)
thresh_size = int(size_of_text * 0.9)
train_data, val_data = text[:thresh_size], text[thresh_size:]

train_ids = gpt2_encoder.encode_ordinary(train_data)
train_ids = np.array(train_ids, dtype=np.int32)
train_ids = torch.from_numpy(train_ids).to(DEVICE)

val_ids = gpt2_encoder.encode_ordinary(val_data)
val_ids = np.array(val_ids, dtype=np.int32)
val_ids = torch.from_numpy(val_ids).to(DEVICE)


def generate_batch(data_s: torch.Tensor):
    shape = (train_params.batch_size,)
    start_indices_b = torch.randint(0, len(data_s) - model_params.words, shape).to(DEVICE)
    end_indices_b = start_indices_b + model_params.words

    shape = (train_params.batch_size, model_params.words)
    in_sequence_bw = torch.zeros(shape, dtype=torch.int64, device=DEVICE)
    out_sequence_bw = torch.zeros(shape, dtype=torch.int64, device=DEVICE)

    for i in range(train_params.batch_size):
        start, end = start_indices_b[i], end_indices_b[i]
        in_sequence_bw[i] = data_s[start:end]
        out_sequence_bw[i] = data_s[start + 1:end + 1]

    return in_sequence_bw, out_sequence_bw

def generate_epoch() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(train_params.number_of_batches):
        yield generate_batch(train_ids)

# ----------------- TRAINING ----------------- #

with exception_controlled_run() as run:

    # ----------------- INIT SETTINGS ----------------- #


    if train_params.loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function {train_params.loss_function}")

    # ----------------- LOAD MODEL ----------------- #
    last_epoch: float | None = mlflow.get_run(run.info.run_id).data.metrics.get('epoch', None)
    if last_epoch is not None:
        last_epoch = int(last_epoch)
        logger.debug("Last epoch is %s", last_epoch)
        model_uri = f"runs:/{mlflow_settings.run_id}/nanogpt_{last_epoch}"
        model: MetricTensorNetwork = mlflow.pytorch.load_model(model_uri)
        model: MetricTensorNetwork = model.to(DEVICE)
        start_epoch = last_epoch + 1
        logger.info(f"Loaded model from {model_uri}")
    else:
        model = MetricTensorNetwork(model_params)
        model.to(DEVICE)
        start_epoch = 0

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=train_params.learning_rate)

    mlflow.log_param("n_parameters", sum(p.numel() for p in parameters))

    # ----------------- TRAINING LOOP ----------------- #

    @contextmanager
    def zero_grad(optimizer: torch.optim.Optimizer) -> Iterator[None]:
        """ Ensures that I don't forget to zero the gradients :)"""

        optimizer.zero_grad()
        yield
        optimizer.step()


    for epoch in range(start_epoch, train_params.number_of_epochs):
        data_gen = generate_epoch()
        data_gen_progressbar = tqdm(data_gen, desc=f"Epoch {epoch}", leave=True)
        for in_sequence_bw, out_sequence_bw in data_gen_progressbar:
            in_sequence_bw.to(DEVICE)
            out_sequence_bw.to(DEVICE)

            
            with zero_grad(optimizer):
                pred_logits_bwt = model(in_sequence_bw)
                # cross entropy expects (batch, classes, sequence)
                pred_logits_btw = pred_logits_bwt.transpose(-1, -2)
                loss = loss_function(pred_logits_btw, out_sequence_bw)
                loss.backward()
    
            data_gen_progressbar.set_postfix(loss=loss.item())

            if mlflow_settings.is_local:
                continue

            if instance_will_terminate():
                raise PauseTraining                   

        # Save the model and log the loss

        mlflow.pytorch.log_model(model, f"mtn_{epoch}")
        mlflow.log_metric("loss", loss.item(), epoch)
        mlflow.log_metric("epoch", epoch, epoch)
 
    


