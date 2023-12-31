import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
from tqdm import tqdm
from model import NanoGPT
import torch
import logging
import os
from contextlib import contextmanager
from train_config import TrainConfiguration, ModelHandler, MLFlowHandler, DEVICE, PauseRunException
import argparse
import requests

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

RUN_ID = os.environ.get("RUN_ID", None)
if not RUN_ID and not args.create_run:
    raise ValueError("RUN_ID environment variable is not set")


def check_termination_notice():
    """
    Checks for EC2 Spot Instance termination notice
    Returns True if termination notice is received, False otherwise
    """
    try:
        response = requests.get('http://169.254.169.254/latest/meta-data/spot/termination-time')
        response.raise_for_status()
        return True
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("Something went wrong",err)
    return False

if args.is_local_run:
    check_termination_notice = lambda : False

with MLFlowHandler.continue_or_create_run(RUN_ID) as mlflow_handler:
    logger.info(f"MLFlow run status is {mlflow_handler.get_status()}")

    if not RUN_ID:
        model_params = ModelHandler()
        train_params = TrainConfiguration()
        model_params.save_to_mlflow()
        train_params.save_to_mlflow()
        del model_params
        del train_params

    model_params = ModelHandler.load_from_mlflow()
    train_params = TrainConfiguration.load_from_mlflow()
    logging.info(f"Train parameters are {train_params}")    
    logging.info(f"Model parameters are {model_params}")

    nanoGPT = NanoGPT(model_params).to(DEVICE)
    run_id = mlflow.active_run().info.run_id
    run = mlflow.get_run(run_id)
    steps = run.data.metrics.get('step', [])

    if steps:
        last_epoch = max(steps)
        logger.debug("Last epoch is %s", last_epoch)
        model_uri = f"runs:/{RUN_ID}/gpt_array_sorter_epoch_{last_epoch}"
        state_dict = mlflow.pytorch.load_model(model_uri)
        nanoGPT.load_state_dict(state_dict)
        start_epoch = last_epoch + 1
        logger.info(f"Loaded model from {model_uri}")
    else:
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

    for epoch in range(start_epoch, train_params.number_of_epochs):
        data_generator = generate_data()
        progress_bar = tqdm(data_generator, desc=f"Epoch {epoch}", leave=True)
        for in_sequence_bw, true_out_sequence_bw in progress_bar:
            with zero_grad(optimizer):
                out_sequence_bw = nanoGPT(in_sequence_bw)
                out_sequence_wb = out_sequence_bw.transpose(-1, -2)
                loss = loss_function(out_sequence_wb, true_out_sequence_bw)
                loss.backward()
            progress_bar.set_postfix(loss=loss.item())
            if check_termination_notice():
                raise PauseRunException()

        mlflow.pytorch.log_model(nanoGPT, f"gpt_array_sorter_epoch_{epoch}")
        mlflow.log_metric("loss", loss.item(), epoch)
