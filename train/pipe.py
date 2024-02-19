# type: ignore

from prefect import task, flow, get_run_logger
import torch
from torch import Tensor
import gc
import numpy as np
from model import ModelFactory, SentimentAnalysisModel
from core.constants import DEVICE
from torch import nn
from functools import cache
import mlflow
from typing import Iterator


@task
def cleanup_memory() -> None:
    """ Clean up memory. """

    logger = get_run_logger()
    gc.collect()
    if not torch.cuda.is_available():
        logger.warning("GPU is not available.")
        return

    torch.cuda.empty_cache()
    logger.info(f"Emptied gpu cache.")


@task
@cache
def load_dataset_from_npz(path: str) -> tuple[Tensor, Tensor]:
    """ Load a dataset from a .npz file. """
    data = np.load(path)
    rating, text = data["rating"], data["text"]
    rating = torch.tensor(rating.astype(np.int64))
    text = torch.tensor(text.astype(np.int32))
    rating = rating[0:300]
    text = text[0:300, :15]
    return rating - 1, text

@task
def load_model(**kwargs) -> SentimentAnalysisModel:
    """ Creates a new model based on the environment variables. """
    logger = get_run_logger()
    model_factory = ModelFactory(**kwargs)
    model = SentimentAnalysisModel(model_factory)
    for i in [0, 1]:
        layer = model.transformer[i]
        layer_name = layer.__class__.__name__
        layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        logger.info(f"Layer {layer_name} has {layer_params} trainable parameters")

    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} learnable parameters.")
    return model.to(DEVICE)

def halt() -> None:
    import time
    time.sleep(10000)


@task
def forward_pass(
    model: SentimentAnalysisModel,
    loss_function: callable,
    rating_b: Tensor,
    text_bw: Tensor,
) -> Tensor:
    """ Forward pass through the model. """
    from transformers import AutoTokenizer
    import tiktoken


    gpt2_encoder = tiktoken.get_encoding("gpt2")
    text_bw = gpt2_encoder.decode_batch(text_bw.tolist())
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    text_bw = tokenizer(
        text_bw,
        return_tensors="pt",
        padding=True,
        truncation=False,
        pad_to_multiple_of=250,
    )
    text_bw = text_bw["input_ids"].to(DEVICE)
    print(text_bw.shape)
    predicted_logits_b5 = model(text_bw)
    loss = loss_function(predicted_logits_b5, rating_b)
    return predicted_logits_b5, loss

@task
def backward_pass(loss_train, multiplier: float = 1.) -> None:
    """ Backward pass through the model. """

    (loss_train * multiplier).backward()

@task(log_prints=False)
def gradient_descent(optimizer: torch.optim.Optimizer) -> None:
    """ Perform a gradient descent step. """

    optimizer.step()
    optimizer.zero_grad()


def yield_batches(
    rating: torch.Tensor,
    text: torch.Tensor,
    gpu_batch_size: int,
    shuffle: bool = True
) -> Iterator[tuple[int, Tensor, Tensor]]:
    """ Yield batches of data. """

    if shuffle:
        random_idx = torch.randperm(len(rating))
        rating = rating[random_idx]
        text = text[random_idx]
        del random_idx

    step = 1
    for start in range(0, len(rating), gpu_batch_size):
        end = start + gpu_batch_size
        rating_batch_b = rating[start:end].to(DEVICE)
        text_batch_bw = text[start:end].to(DEVICE)
        yield step, rating_batch_b, text_batch_bw
        step += 1


@task
@cache
def create_optimizer(model: SentimentAnalysisModel) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters())

@task
def calculate_accuracy(predicted_logits_b5, rating_b):
    predicted_rating_b = torch.argmax(predicted_logits_b5, dim=1)
    return (predicted_rating_b == rating_b).float().mean()

def training_loop(model: SentimentAnalysisModel, loss_function: callable, accumulation_steps = 10):
    optimizer = create_optimizer(model)
    training_data = load_dataset_from_npz("data/asa/test.npz")
    for step, rating_batch_b, text_batch_bw in yield_batches(training_data[0], training_data[1], 15):
        model.train()
        _, loss = forward_pass(model, loss_function, rating_batch_b, text_batch_bw)
        backward_pass(loss, multiplier=1/accumulation_steps)
        if step % accumulation_steps == 0:
            gradient_descent(optimizer)
            yield step, model, loss

def evaluation_loop(model: SentimentAnalysisModel, loss_function: callable) -> Iterator[tuple[int, float, float]]:
    model.eval()
    test_data = load_dataset_from_npz("data/asa/test.npz")
    with torch.no_grad():
        for step, rating_batch_b, text_batch_bw in yield_batches(test_data[0], test_data[1], 15):
            predicted_logits_b5, loss = forward_pass(model, loss_function, rating_batch_b, text_batch_bw)
            accuracy = calculate_accuracy(predicted_logits_b5, rating_batch_b)
            yield step, loss, accuracy

@task
def calculate_stats(metric_name: str, metric_values: np.ndarray):
    return {
        f"{metric_name}/count": len(metric_values),
        f"{metric_name}/mean": np.mean(metric_values),
        f"{metric_name}/median": np.median(metric_values),
        f"{metric_name}/std": np.std(metric_values),
        f"{metric_name}/var": np.var(metric_values),
        f"{metric_name}/max": np.max(metric_values),
        f"{metric_name}/min": np.min(metric_values),

        f"{metric_name}/p95": np.percentile(metric_values, 95),
        f"{metric_name}/p90": np.percentile(metric_values, 90),
        f"{metric_name}/p75": np.percentile(metric_values, 75),
        f"{metric_name}/p25": np.percentile(metric_values, 25),
        f"{metric_name}/p10": np.percentile(metric_values, 10),
        f"{metric_name}/p5": np.percentile(metric_values, 5),
    }

@flow
def perform_evaluation(model, loss_function, total_count: int = 10):

    metrics = {
        "loss": [],
        "accuracy": [],
    }

    for _, loss, accuracy in evaluation_loop(model, loss_function):
        metrics["loss"].append(loss.item())
        metrics["accuracy"].append(accuracy.item())

        if len(metrics["loss"]) == total_count:
            return metrics

    assert False


@flow
def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("asa")
    model = load_model(
        words = 250,
        coordinates = 10,
        number_of_heads = 1,
    )

    loss_function = nn.CrossEntropyLoss()

    eval_interval = 1
    save_interval = 1
    num_epochs = 100
    accumulation_steps = 1

    number_of_gradient_descent_steps = 0

    for epoch_step in range(num_epochs):
        cleanup_memory()
        
        for training_loop_step, model, loss in training_loop(
            model, loss_function, accumulation_steps=accumulation_steps
        ):
            number_of_gradient_descent_steps += 1

            # observability into the training loop
    
            mlflow.log_metrics(
                {
                    "epoch": epoch_step,
                    "loss/train": loss.item(),
                    "lr": .1e-3,
                }, 
                step=number_of_gradient_descent_steps,
                synchronous=False
            )

            if training_loop_step % eval_interval == 0:
                metrics = perform_evaluation(model, loss_function, total_count=5)
                for metric_name, metric_values in metrics.items():
                    mlflow.log_metrics(
                        calculate_stats(metric_name, metric_values),
                        step=number_of_gradient_descent_steps,
                        synchronous=False
                    )
    
            if training_loop_step % save_interval == 0:
                mlflow.pytorch.log_model(model, "models")

if __name__ == "__main__":
    main()
