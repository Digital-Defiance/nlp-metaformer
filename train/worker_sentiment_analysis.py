import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import torch
from train.config import TrainingLoopFactory, MLFlowSettings
from model import ModelFactory
import torch
from core.logger import get_logger
from core.constants import DEVICE
import pickle

def progressbar_range(total_size, batch_size: int, epoch: int):
    return tqdm(
        range(0, total_size, batch_size),
        desc=f"({epoch})",
        leave=True
    )

# ----------------- SETTINGS ----------------- #

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)
logger.info(f"Using device {DEVICE}")
logger.info(f"Using torch version {torch.__version__}")
logger.info(f"Using mlflow version {mlflow.__version__}")

mlflow_settings = MLFlowSettings()
training_loop_factory = TrainingLoopFactory()
model_factory =  ModelFactory(

)


def reshuffle_batches(x, y):
    random_idx = torch.randperm(len(x))
    return x[random_idx], y[random_idx]

def process_split(sentences, sentiments):
    sentiments += 50258
    padding = sentences != -1 # 50257
    gt_sentences = torch.clone(sentences)
    gt_sentences[padding] = 0
    return gt_sentences + padding.long() * sentiments.unsqueeze(1)

with open("stanfordSentimentTreebank.pickle", "rb") as f:
    dataset = pickle.load(f)
    dev_sentences, dev_sentiments = dataset['dev']
    train_sentences, train_sentiments = dataset['train']
    del dataset

train_sentences, train_sentiments, dev_sentences, dev_sentiments = map(
    lambda x: torch.tensor(x).to(DEVICE),
    [train_sentences, train_sentiments, dev_sentences, dev_sentiments],
)
size = len(train_sentences)
train_gt_sentences = process_split(train_sentences, train_sentiments)
dev_gt_sentences = process_split(dev_sentences, dev_sentiments)
del train_sentiments, dev_sentiments



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
    for epoch in range(1, training_loop_factory.number_of_epochs + 1):
        lr = get_lr(epoch)
        optimizer.set_lr(lr)
        mlflow.log_metric("lr", lr, step=epoch)

        train_sentences, train_gt_sentences = reshuffle_batches(train_sentences, train_gt_sentences)

        training_loss_cumul = 0
        counter = 0
        last_count = len(train_sentences) // training_loop_factory.batch_size

        for step in (
            pb := progressbar_range(
                len(train_sentences),
                training_loop_factory.batch_size,
                epoch,
            )
        ):
            counter += 1
     
            train_sentence_bw = train_sentences[step:step+training_loop_factory.batch_size]
            train_gt_sentence_bw = train_gt_sentences[step:step+training_loop_factory.batch_size]


            # Perform feed forward + backwards propagation + gradient descent
            optimizer.zero_grad()
            pred_logits_bwt = model(train_sentence_bw)
            pred_logits_btw = pred_logits_bwt.transpose(-1, -2)
            loss_train = loss_function(pred_logits_btw, train_gt_sentence_bw)
            loss_train.backward()
            optimizer.step()

            # Log the training loss
            training_loss_cumul += loss_train.item()
    

            to_log = {
                "train": training_loss_cumul / counter,
                "lr": f"{lr:.2e}"
            }
        
            if counter != last_count:
                pb.set_postfix(to_log)
                continue

            with torch.no_grad():
                val_loss_cumul = 0
                val_counter = 0
                for val_step in range(0, len(dev_sentences), training_loop_factory.batch_size):
                    val_counter += 1
                    dev_sentence_bw = dev_sentences[step:step+training_loop_factory.batch_size]
                    dev_gt_sentence_bw = dev_gt_sentences[step:step+training_loop_factory.batch_size]
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
                "loss/val": to_log["val"],
                "epoch": epoch,
            },
            step=epoch,
        )
 



