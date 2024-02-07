import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from torch import nn
from pydantic_settings import BaseSettings
from typing import  Optional, Literal, Dict, Any
from core.mixins import MyBaseSettingsMixin
from core.logger import get_logger
from core.constants import DEVICE
from model import ModelFactory, SentimentAnalysisModel
from data.worker import Worker

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")


worker = Worker()
model_factory =  ModelFactory()
task = worker.request_data(0, model_factory.words)



class TrainSettings(BaseSettings, MyBaseSettingsMixin):
    number_of_epochs: int = 100
    batch_size: int = 32
    number_of_slices: int = 2
    l1_regularization: float = 0
    l2_regularization: float = 0
    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-9
    warmup_steps: int = 4000

    class Config:
        env_prefix = "TRAIN_"

class Adam(torch.optim.Adam):
    def set_lr(self, lr: float):
        for param_group in self.param_groups:
            param_group['lr'] = lr





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


# MLFLOW_TRACKING_URI=http://mlflow:80

mlflow_settings = MLFlowSettings()
train_settings = TrainSettings()

with mlflow.start_run(**mlflow_settings.model_dump()) as run:
    logger.info("Connected to MLFlow and started run.")

    model = SentimentAnalysisModel().to(DEVICE)
    logger.info(f"Created model and moved it to {DEVICE}")


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_factory.save_to_mlflow()
    train_settings.save_to_mlflow()
    mlflow.log_param("number_of_parameters", n_parameters)
    logger.info("Saved training info and hyperparameters to MLFLow")
    logger.info(f"Model has {n_parameters} parameters")
    del n_parameters


    optimizer = Adam(
        model.parameters(),
        lr=1,
        betas=(train_settings.beta_1, train_settings.beta_2),
        eps=train_settings.epsilon,
    )

    loss_function = nn.CrossEntropyLoss()

    step: int = 0
    for epoch in range(1, train_settings.number_of_epochs + 1):
        for epoch_slice_idx in range(train_settings.number_of_slices):

            logger.info(f"Fetching slice {epoch_slice_idx} from worker...")
            rating, text = task.get()
            logger.info("Fetched slice from worker.")
            task.forget()
            logger.info("Deleted slice from redis.")
            rating, text = torch.tensor(rating).float(), torch.tensor(text)
            random_idx = torch.randperm(len(rating))
            rating, text = rating[random_idx], text[random_idx]
            del random_idx


            task = worker.request_data(epoch_slice_idx, model_factory.words)
            logger.info(f"Schedule slice {epoch_slice_idx}.")
            logger.info("Starting training loop...")
            metrics = {
                "epoch": epoch,
                "slice_idx": epoch_slice_idx,
            }

            for i in tqdm(
                range(len(rating) // train_settings.batch_size),
                desc=f"Epoch {epoch}, Slice {epoch_slice_idx})",
                leave=True,
            ):
                step += 1

                # Handle learning rate
                metrics["lr"] = min(
                    step ** -0.5,
                    step * train_settings.warmup_steps ** -1.5
                ) * model_factory.coordinates ** -0.5
                metrics["lr"] = metrics["lr"] / 10
                optimizer.set_lr(metrics["lr"])

                # Create batch from the slice
                start = i*train_settings.batch_size
                end = start + train_settings.batch_size
                rating_batch_b = rating[start:end].to(DEVICE)
                text_batch_bw = text[start:end].to(DEVICE)

                # Perform feed forward + backwards propagation + gradient descent
                optimizer.zero_grad()
                pred_logits_b5 = model(text_batch_bw)
                loss_train = loss_function(pred_logits_b5, rating_batch_b)
                loss_train.backward()
                optimizer.step()

                # Log the training loss
                metrics["loss/train"] = loss_train.item()
                mlflow.log_metrics(metrics, step=step)
        

            """
            with torch.no_grad():
                val_loss_cumul = 0
                val_counter = 0
                for val_step in range(0, len(dev_sentences), train_settings.batch_size):
                    val_counter += 1
                    dev_sentence_bw = dev_sentences[val_step:val_step+train_settings.batch_size]
                    dev_gt_sentence_bw = dev_gt_sentences[val_step:val_step+train_settings.batch_size]
                    pred_logits_bwt = model(dev_sentence_bw)
                    pred_logits_btw = pred_logits_bwt.transpose(-1, -2)
                    loss_val = loss_function(pred_logits_btw, dev_gt_sentence_bw)
                    val_loss_cumul += loss_val.item()
                to_log["val"] = val_loss_cumul / val_counter
                pb.set_postfix(to_log)

            # mlflow.pytorch.log_model(model, f"mtn_{epoch}")
            mlflow.log_metrics(
                {
                    "loss/train": to_log["train"],
                    # "loss/val": to_log["val"],
                    "epoch": epoch,
                },
                step=epoch,
            )
            """

 



