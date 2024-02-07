import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import gc

from pydantic_settings import BaseSettings
from typing import  Optional, Literal, Dict, Any
from core.mixins import MyBaseSettingsMixin
from core.logger import get_logger
from core.constants import DEVICE
from model import ModelFactory
from data.worker import Worker
from train.config import TrainingLoopFactory

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")


worker = Worker()
model_factory =  ModelFactory()
task = worker.request_data(0, model_factory.words)


class SentimentAnalysisModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        model = model_factory.create_model(kind="encoder")
        del model[-1]
        self.transformer = model
        self.project_context = torch.nn.Linear(model_factory.words, 5)
        self.project_coordinates = torch.nn.Linear(model_factory.coordinates, 1)

    def forward(self, x_bw):
        x_bwc = self.transformer(x_bw)
        x_bw = self.project_coordinates(x_bwc)[:, :, 0]
        x_b5 = self.project_context(x_bw)
        return x_b5


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
training_loop_factory = TrainingLoopFactory()

with mlflow.start_run(**mlflow_settings.model_dumpt()) as run:
    logger.info("Connected to MLFlow and started run.")

    model = SentimentAnalysisModel().to(DEVICE)
    logger.info(f"Created model and moved it to {DEVICE}")


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_factory.save_to_mlflow()
    training_loop_factory.save_to_mlflow()
    logger.info("Saved training info and hyperparameters to MLFLow")
    logger.info(f"Model has {n_parameters} parameters")
    
    

    optimizer = training_loop_factory.create_optimizer(model.parameters())
    get_lr = training_loop_factory.create_scheduler(model_factory.coordinates)
    loss_function = training_loop_factory.create_loss_function()

    step: int = 0
    for epoch in range(1, training_loop_factory.number_of_epochs + 1):
        for epoch_slice_idx in range(2):

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
            for i in tqdm(
                range(len(rating) // training_loop_factory.batch_size),
                desc=f"Epoch {epoch}, Slice {epoch_slice_idx})",
                leave=True,
            ):
                step += 1
                lr = get_lr(step) / 10
                optimizer.set_lr(lr)
                start = i*training_loop_factory.batch_size
                end = start + training_loop_factory.batch_size
        
                rating_batch_b5 = rating[start:end].to(DEVICE)
                text_batch_bw = text[start:end].to(DEVICE)

                # Perform feed forward + backwards propagation + gradient descent
                optimizer.zero_grad()
                pred_logits_b5 = model(text_batch_bw)
                loss_train = loss_function(pred_logits_b5, rating_batch_b5)
                loss_train.backward()
                optimizer.step()

                # Log the training loss
                metrics = {
                    "loss/train": loss_train.item(),
                    "epoch": epoch,
                    "slice_idx": epoch_slice_idx,
                    "lr": lr,
                }

                mlflow.log_metrics(metrics, step=step)
        

            """
            with torch.no_grad():
                val_loss_cumul = 0
                val_counter = 0
                for val_step in range(0, len(dev_sentences), training_loop_factory.batch_size):
                    val_counter += 1
                    dev_sentence_bw = dev_sentences[val_step:val_step+training_loop_factory.batch_size]
                    dev_gt_sentence_bw = dev_gt_sentences[val_step:val_step+training_loop_factory.batch_size]
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

 



