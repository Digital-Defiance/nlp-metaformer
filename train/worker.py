import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import torch
from contextlib import contextmanager
from train.config import TrainingLoopFactory, MLFlowSettings
from model import ModelFactory
from mlflow.entities import RunStatus
import requests
import torch
from typing import Iterator
from core.logger import get_logger
from core.constants import DEVICE


# ----------------- SETTINGS ----------------- #

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")

mlflow_settings = MLFlowSettings()
training_loop_factory = TrainingLoopFactory()
model_factory = ModelFactory()

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
        "experiment_id": 5,
        "log_system_metrics": mlflow_settings.log_system_metrics,
    }

    logger.info(f"Starting run with kwargs {run_kwargs}")

    def set_status(status: RunStatus):
        status = RunStatus.to_string(status)
        mlflow.tracking.MlflowClient().set_terminated(mlflow_settings.run_id, status=status)

    try:
        with mlflow.start_run(**run_kwargs) as run:
            mlflow_settings.run_id = run.info.run_id
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



# ----------------- TRAINING ----------------- #

with exception_controlled_run() as run:

    # ----------------- LOAD MODEL ----------------- #

    if mlflow_settings.has_checkpoint():
        model, epoch = mlflow_settings.load_model()
        start_epoch = epoch + 1
    else:
        model = model_factory.create_model()
        model_factory.save_to_mlflow()
        training_loop_factory.save_to_mlflow()
        start_epoch = 0


    parameters = model.parameters()
    mlflow.log_param("n_parameters", sum(p.numel() for p in parameters))
    
    optimizer = training_loop_factory.create_optimizer(model.parameters())
    loss_function = training_loop_factory.create_loss_function()
    create_training_batch, create_validation_batch, create_epoch_data = training_loop_factory.create_data_handlers()


    # ----------------- TRAINING LOOP ----------------- #

    @contextmanager
    def zero_grad(optimizer: torch.optim.Optimizer) -> Iterator[None]:
        """ Ensures that I don't forget to zero the gradients :)"""

        optimizer.zero_grad()
        yield
        optimizer.step()


    for epoch in range(start_epoch, training_loop_factory.number_of_epochs):

        data_gen = create_epoch_data()

        for in_sequence_bw, out_sequence_bw in tqdm(data_gen, desc=f"Epoch {epoch}", leave=True):
            with zero_grad(optimizer):
                pred_logits_bwt = model(in_sequence_bw)
                # cross entropy expects (batch, classes, sequence)
                pred_logits_btw = pred_logits_bwt.transpose(-1, -2)
                loss_train = loss_function(pred_logits_btw, out_sequence_bw)
                loss_train.backward()

            if mlflow_settings.is_local:
                continue

            if instance_will_terminate():
                raise PauseTraining                   


        with torch.no_grad():
            in_sequence_bw, out_sequence_bw = create_validation_batch()
            pred_logits_bwt = model(in_sequence_bw)
            # cross entropy expects (batch, classes, sequence)
            pred_logits_btw = pred_logits_bwt.transpose(-1, -2)
            loss_val = loss_function(pred_logits_btw, out_sequence_bw)


        mlflow.pytorch.log_model(model, f"mtn_{epoch}")
        mlflow.log_metric("loss/train", loss_train.item(), epoch)
        mlflow.log_metric("loss/val", loss_val.item(), epoch)
        mlflow.log_metric("epoch", epoch, epoch)
 
    


