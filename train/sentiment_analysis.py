import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import gc

from core.logger import get_logger
from core.constants import DEVICE
from model import ModelFactory
from data.worker import Worker
from train.config import TrainingLoopFactory, MLFlowSettings



worker = Worker()
task = worker.request_data(0)




# ----------------- SETTINGS ----------------- #

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")

mlflow_settings = MLFlowSettings()
training_loop_factory = TrainingLoopFactory()
model_factory =  ModelFactory()


def reshuffle_batches(x, y):
    random_idx = torch.randperm(len(x))
    return x[random_idx], y[random_idx]




with mlflow.start_run(
    run_id=mlflow_settings.run_id,
    experiment_id=mlflow_settings.experiment_id,
    log_system_metrics=mlflow_settings.log_system_metrics,
) as run:
    
    logger.info("Connected to MLFlow and started run.")

    model = model_factory.create_model(kind="encoder")
    logger.info("Created model")

    model_factory.save_to_mlflow()
    logger.info("Saved model info to MLFLow")

    training_loop_factory.save_to_mlflow()
    logger.info("Saved training info to MLFLow")

    mlflow.log_param("n_parameters",  sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = training_loop_factory.create_optimizer(model.parameters())
    get_lr = training_loop_factory.create_scheduler(model_factory.coordinates)
    loss_function = training_loop_factory.create_loss_function()


    logger.info("Starting training loop")
    step = 0
    for epoch in range(1, training_loop_factory.number_of_epochs + 1):
        lr = get_lr(epoch) / 10
        optimizer.set_lr(lr)
        mlflow.log_metric("lr", lr, step=epoch)


        rating, text = task.get()
        rating, text = torch.tensor(rating).to(DEVICE), torch.tensor(text).to(DEVICE)
        gc.collect()
        rating, text = reshuffle_batches(rating, text)
        epoch_slice = 0
        task = worker.request_data(epoch_slice)

        for start in (
            pb := tqdm(
                range(len(rating) // training_loop_factory.batch_size),
                desc=f"({epoch}/{epoch_slice})",
                leave=True,
            )
        ):
            step += 1
            end = start + training_loop_factory.batch_size
    
            rating_batch_bw = rating[start:end]
            text_batch_bw = text[start:end]

            # Perform feed forward + backwards propagation + gradient descent
            optimizer.zero_grad()
            pred_logits_bwt = model(text_batch_bw)
            pred_logits_btw = pred_logits_bwt.transpose(-1, -2)
            loss_train = loss_function(pred_logits_btw, rating_batch_bw)
            loss_train.backward()
            optimizer.step()

            # Log the training loss
            mlflow.log_metric("loss/train", loss_train.item(), step=step)
            mlflow.log_metric("epoch", epoch, step=step)
            mlflow.log_metric("lr", lr, step=step)
            pb.set_postfix({"loss/train": loss_train.item()})

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

 



