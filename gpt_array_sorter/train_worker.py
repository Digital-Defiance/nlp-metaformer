import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
from tqdm import tqdm
from model import NanoGPT
import torch
import logging
from contextlib import contextmanager
from train_config import TrainConfiguration, ModelHandler, DEVICE, MLFlowSettings
import argparse
from typing import Optional
from mlflow.entities import RunStatus
import boto3
from botocore.exceptions import BotoCoreError, BotoConnectionError


logging.getLogger("mlflow").setLevel(logging.DEBUG)
torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")


parser = argparse.ArgumentParser(description='Train the model.')

parser.add_argument(
    '--create',
    dest='create_run',
    action='store_true',
    help='create the run if set, otherwise continue an existing run'
)

parser.add_argument(
    '--local',
    dest='is_local_run',
    action='store_true',
    help='indicates if this is being run in aws or locally'
)

args = parser.parse_args()
mlflow_settings = MLFlowSettings()
mlflow.set_tracking_uri(mlflow_settings.tracking_url)
logger.info(mlflow_settings)

if not mlflow_settings.run_id and not args.create_run:
    raise ValueError("RUN_ID environment variable is not set")



with mlflow.start_run(
    run_id=mlflow_settings.run_id,
    experiment_id=mlflow_settings.experiment_id,
    log_system_metrics=mlflow_settings.log_system_metrics,
) as run:
    logger.info(f"MLFlow run status is {mlflow.get_run(run.info.run_id).info.status}")

    if not mlflow_settings.run_id:
        mlflow_settings.run_id = run.info.run_id
        for cls in [TrainConfiguration, ModelHandler]:
            cls().save_to_mlflow()
        del cls

    train_params = TrainConfiguration.load_from_mlflow()
    model_params = ModelHandler.load_from_mlflow()
    logging.info(f"Train parameters are {train_params}")    
    logging.info(f"Model parameters are {model_params}")
    
    last_epoch = mlflow.get_run(run.info.run_id).data.metrics.get('epoch', None)
    if last_epoch:
        last_epoch = int(last_epoch)
        logger.debug("Last epoch is %s", last_epoch)
        model_uri = f"runs:/{mlflow_settings.run_id}/nanogpt_{last_epoch}"
        nanoGPT = mlflow.pytorch.load_model(model_uri).to(DEVICE)
        start_epoch = last_epoch + 1
        logger.info(f"Loaded model from {model_uri}")
    else:
        nanoGPT = NanoGPT(model_params).to(DEVICE)
        start_epoch = 0

    optimizer = torch.optim.Adam(nanoGPT.parameters(), lr=train_params.learning_rate)

    if train_params.loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function {train_params.loss_function}")

    def generate_data():
        shape = (train_params.number_of_batches, model_params.words,)
        for _ in range(train_params.number_of_batches):    
            sequence_bw = torch.randint(0, model_params.tokens, shape)
            sequence_bw = sequence_bw.to(DEVICE)
            sorted_sequence_bw, _ = torch.sort(sequence_bw, dim=1)
            yield sequence_bw, sorted_sequence_bw

    @contextmanager
    def zero_grad(optimizer):
        optimizer.zero_grad()
        yield
        optimizer.step()

    pause_training = False
    for epoch in range(start_epoch, train_params.number_of_epochs):
        data_gen = generate_data()
        data_gen_progressbar = tqdm(data_gen, desc=f"Epoch {epoch}", leave=True)
        for in_sequence_bw, true_out_sequence_bw in data_gen_progressbar:

            with zero_grad(optimizer):
                out_sequence_bw = nanoGPT(in_sequence_bw)
                out_sequence_wb = out_sequence_bw.transpose(-1, -2)
                loss = loss_function(out_sequence_wb, true_out_sequence_bw)
                loss.backward()
    
            data_gen_progressbar.set_postfix(loss=loss.item())

            if args.is_local_run:
                continue

            try:
                session = boto3.Session()
                metadata = session.client('ec2-instance-metadata')
                termination_time: Optional[str] = metadata.get('spot/termination-time')
                if termination_time:
                    logger.info("Received termination notice")
                    pause_training = True
                    break
            except (BotoCoreError, BotoConnectionError) as error:
                logger.error("Error:", error)


            pause_training = True
            break
        if pause_training:
            break

        mlflow.pytorch.log_model(nanoGPT, f"nanogpt_{epoch}")
        mlflow.log_metric("loss", loss.item(), epoch)
        mlflow.log_metric("epoch", epoch, epoch)

if pause_training:
    status = RunStatus.to_string(RunStatus.SCHEDULED)
    mlflow.tracking.MlflowClient().set_terminated(mlflow_settings.run_id, status=status)
