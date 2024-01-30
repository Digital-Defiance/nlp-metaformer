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

def progressbar_range(n_batches: int, epoch: int):
    return tqdm(
        range(1, n_batches + 1),
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
"""
    coordinates = 30,
    words = 70,
    tokens=50258 + 5,
    number_of_blocks = 3,
    number_of_heads = 3,
    bias = False,
    attention = "metric" # "scaled_dot_product", # or "metric"
"""


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

logger.info("Connecting to MLFlow and starting")

with mlflow.start_run(
    run_id=mlflow_settings.run_id,
    experiment_id=mlflow_settings.experiment_id,
    log_system_metrics=mlflow_settings.log_system_metrics,
) as run:

    logger.info("Loading model")
    model = model_factory.create_model(kind="encoder")
    model_factory.save_to_mlflow()
    training_loop_factory.save_to_mlflow()
    mlflow.log_param("n_parameters",  sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = training_loop_factory.create_optimizer(model.parameters())
    get_lr = training_loop_factory.create_scheduler(model_factory.coordinates)
    loss_function = training_loop_factory.create_loss_function()
    data_factory = training_loop_factory.create_data_factory()


    for epoch in range(1, training_loop_factory.number_of_epochs + 1):
        lr = get_lr(epoch)
        optimizer.set_lr(lr)
        mlflow.log_metric("lr", lr, step=epoch)

        train_sentences, train_gt_sentences = reshuffle_batches(train_sentences, train_gt_sentences)

        training_loss_cumul = 0

        for step in (pb := progressbar_range(training_loop_factory.number_of_batches, epoch)):
     
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
    
            pb.set_postfix({
                "train": training_loss_cumul / (step + 1),
                "lr": f"{lr:.2e}"
            })

        with torch.no_grad():
            in_sequence_bw, out_sequence_bw = data_factory.create_batch(
                split="validation",
                max_size_of_sequence=model_factory.words,
            )
            pred_logits_bwt = model(in_sequence_bw)
            pred_logits_btw = pred_logits_bwt.transpose(-1, -2)
            loss_val = loss_function(pred_logits_btw, out_sequence_bw)


        mlflow.pytorch.log_model(model, f"mtn_{epoch}")
        mlflow.log_metrics(
            {
                "loss/train": loss_train.item(),
                "loss/val": loss_val.item(),
                "epoch": epoch,
            },
            step=epoch,
        )
 



