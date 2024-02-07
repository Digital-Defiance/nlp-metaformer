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
model_factory =  ModelFactory(
    words=100,
    tokens=6,
    coordinates=6,
    number_of_heads=3,

)



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


# def get():
#    return torch.randint(0, 5, (64, 5)), torch.randint(0, 5, (64, 100))

# y, x = get()




task = worker.request_data(0, model_factory.words)





class SentimentAnalysisModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.transformer = model_factory.create_model(kind="encoder")
        self.project_tokens = torch.nn.Sequential(
            torch.nn.Linear(model_factory.tokens, model_factory.tokens // 2),
            torch.nn.Linear(model_factory.tokens // 2, model_factory.tokens // 4),
            torch.nn.Linear(model_factory.tokens // 4, 5),
        )
        self.project_context = torch.nn.Linear(model_factory.words, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.project_tokens(x)
        x = self.project_context(x.transpose(-1, -2))[:, :, 0]
        return x


    




# ----------------- SETTINGS ----------------- #

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")

mlflow_settings = MLFlowSettings()
training_loop_factory = TrainingLoopFactory()


with mlflow.start_run(
    run_id=mlflow_settings.run_id,
    experiment_id=mlflow_settings.experiment_id,
    log_system_metrics=mlflow_settings.log_system_metrics,
) as run:
    logger.info("Connected to MLFlow and started run.")

    model = SentimentAnalysisModel().to(DEVICE)
    logger.info("Created model")

    model_factory.save_to_mlflow()
    logger.info("Saved model info to MLFLow")

    training_loop_factory.save_to_mlflow()
    logger.info("Saved training info to MLFLow")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_param("n_parameters", n_parameters)
    logger.info(f"Model has {n_parameters} parameters")

    optimizer = training_loop_factory.create_optimizer(model.parameters())
    get_lr = training_loop_factory.create_scheduler(model_factory.coordinates)
    loss_function = training_loop_factory.create_loss_function()

    logger.info("Starting training loop...")
    step = 0
    for epoch in range(1, training_loop_factory.number_of_epochs + 1):

        


        rating, text = task.get()
        task.forget()
        # rating, text = get()
        rating, text = torch.tensor(rating), torch.tensor(text)
        gc.collect()
        random_idx = torch.randperm(len(rating))
        rating, text = rating[random_idx], text[random_idx]
        del random_idx


        epoch_slice = 0

        task = worker.request_data(epoch_slice, model_factory.words)

        for i in (
            pb := tqdm(
                range(len(rating) // training_loop_factory.batch_size),
                desc=f"({epoch}/{epoch_slice})",
                leave=True,
            )
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
            loss_train = loss_function(pred_logits_b5, rating_batch_b5.float())
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

 



