import torch
import mlflow
from torch import nn
from pydantic_settings import BaseSettings
from typing import  Optional, Literal, Dict, Any
from core.mixins import MyBaseSettingsMixin
from core.logger import get_logger
from core.constants import DEVICE
from model import ModelFactory, SentimentAnalysisModel
from data.worker import request_data, request_task_cleanup
from mlflow import log_metrics, start_run, log_param
import mlflow
import gc
import os

spark_seed = torch.randint(low=1, high=10_000, size=(1,)).item()
torch_seed = torch.randint(low=1, high=10_000, size=(1,)).item()
torch.manual_seed(torch_seed)


torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")

model_factory =  ModelFactory()
task = request_data(0, model_factory.words, spark_seed)
logger.info(f"Requested slice 0")


class Adam(torch.optim.Adam):
    def set_lr(self, lr: float) -> None:
        for param_group in self.param_groups:
            param_group['lr'] = lr

class TrainSettings(BaseSettings, MyBaseSettingsMixin):
    number_of_epochs: int = 100
    batch_size: int = 2
    number_of_slices: int = 2
    l1_regularization: float = 0
    l2_regularization: float = 0
    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-9
    warmup_steps: int = 4000
    lr_schedule_scaling: float = 1

    class Config:
        env_prefix = "TRAIN_"

class MLFlowSettings(BaseSettings, MyBaseSettingsMixin):
    run_id: Optional[str] = None
    experiment_id: int = 1
    run_name: Optional[str] = None
    nested: bool = False
    tags: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    log_system_metrics: Optional[bool] = True

    class Config:
        env_prefix = "MLFLOW_"


mlflow_settings = MLFlowSettings()
train_settings = TrainSettings()

def get_lr(step: int) -> float:
    lr = min(step ** -0.5, step * train_settings.warmup_steps ** -1.5)
    lr = lr * model_factory.coordinates ** -0.5
    lr = min(lr, 1e-3)
    return lr * train_settings.lr_schedule_scaling

with start_run(**mlflow_settings.model_dump()) as run:
    logger.info("Connected to MLFlow and started run.")
    log_param("spark_seed", spark_seed)
    log_param("torch_seed", torch_seed)

    model = SentimentAnalysisModel(model_factory).to(DEVICE)
    logger.info(f"Created model and moved it to {DEVICE}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_factory.save_to_mlflow()
    train_settings.save_to_mlflow()
    log_param("number_of_parameters", n_parameters)
    logger.info("Saved training info and hyperparameters to MLFLow")
    logger.info(f"Model has {n_parameters} parameters")
    del n_parameters

    optimizer = Adam(model.parameters(), lr=1, betas=(train_settings.beta_1, train_settings.beta_2), eps=train_settings.epsilon)
    loss_function = nn.CrossEntropyLoss()
    step: int = 1

    for epoch in range(1, train_settings.number_of_epochs + 1):
        rating, text = None, None
        for epoch_slice_idx in range(train_settings.number_of_slices):
            
            logger.info(f"Cleaning up memory...")
            del rating, text
            gc.collect()
            logger.info(f"Called garbage collector.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"Emptied gpu cache.")
    
            logger.info(f"Fetching slice {epoch_slice_idx} from worker...")
            rating_b_path, text_bw_path = task.get()
            rating = np.load(rating_b_path)
            text = np.load(text_bw_path)
            os.remove(rating_b_path)
            os.remove(text_bw_path)
            rating, text = torch.tensor(rating), torch.tensor(text)
            random_idx = torch.randperm(len(rating))
            rating, text = rating[random_idx], text[random_idx]
            del random_idx

            task = request_data(epoch_slice_idx, model_factory.words, spark_seed)
            logger.info(f"Scheduled slice {epoch_slice_idx}, task id is {task.id}.")

            metrics: dict[str, str | int ] = {
                "epoch": epoch,
                "slice_idx": epoch_slice_idx,
            }

            slice_size = len(rating)
            logger.info(f"Epoch {epoch}, Slice {epoch_slice_idx}")
            for start in range(0, slice_size, 32):
                # Create batch from the slice
                end = start + 32
                rating_batch_b = rating[start:end].to(DEVICE)
                text_batch_bw = text[start:end].to(DEVICE)

                # Perform feed forward + backwards propagation + gradient descent
                pred_logits_b5 = model(text_batch_bw)
                loss_train = loss_function(pred_logits_b5, rating_batch_b)
                (loss_train / train_settings.batch_size).backward()

                if (end // 32) % train_settings.batch_size == 0 or end == slice_size:
                    metrics["loss/train"] = loss_train.item()
                    metrics["lr"] = get_lr(step)
                    optimizer.set_lr(metrics["lr"])
                    optimizer.step()
                    optimizer.zero_grad()
                    log_metrics(metrics, step=step)
                    step += 1
                   
