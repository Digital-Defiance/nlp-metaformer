from train.sentiment_analysis_v2 import training_loop, TrainSettings, ModelFactory, SentimentAnalysisModel, yield_batches, train_step, eval_model
from train.sentiment_analysis_v2 import load_data
import numpy as np
import torch

import pytest


@pytest.fixture(scope="session")
def data():
    return load_data()

@pytest.fixture(scope="session")
def rating(data):
    return data[0][0:500]

@pytest.fixture(scope="session")
def text(data):
    return data[1][0:500, :5]


@pytest.fixture
def model():
    model_factory = ModelFactory(
        coordinates=6,
        tokens=60_000,
        words=5,
        number_of_blocks=1,
        number_of_heads=3,
        bias=False,
        attention="metric",
    )

    return SentimentAnalysisModel(model_factory)

@pytest.fixture
def train_settings():
    return TrainSettings(
        number_of_epochs=2,
        gpu_batch_size=3,
        warmup_steps=1,
        lr_schedule_scaling=1,
        torch_seed=1,
        spark_seed=1,
        number_of_slices=1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )


@pytest.fixture
def get_lr():
    return lambda step: 0.01


def test_train_step(rating, text, model):
    optimizer = torch.optim.Adam(model.parameters())

    model, loss = train_step(model, rating, text, optimizer)

    assert isinstance(model, SentimentAnalysisModel)
    assert model.training == True
    assert isinstance(loss, float)
    assert loss > 0


def test_yield_batches(rating, text, train_settings):
    for rating_batch, text_batch in yield_batches(rating, text, train_settings.gpu_batch_size):
        assert isinstance(rating_batch, torch.Tensor)
        assert isinstance(text_batch, torch.Tensor)

        assert rating_batch.shape[0] <= train_settings.gpu_batch_size
        assert text_batch.shape[0] <= train_settings.gpu_batch_size
        assert rating_batch.shape[0] == text_batch.shape[0]
        assert text_batch.shape[1] == text.shape[1]

        assert rating_batch.dtype == rating.dtype
        assert text_batch.dtype == text.dtype

        

def test_training_loop(rating, text, model, train_settings):

    def get_lr(step: int) -> float:
        assert isinstance(step, int)
        return 10*step

    for model, metrics, step in training_loop(train_settings, model, rating, text, get_lr):
        assert isinstance(model, SentimentAnalysisModel)
        assert isinstance(metrics, dict)
        assert isinstance(step, int)

        assert "epoch" in metrics
        assert "lr" in metrics
        assert metrics["lr"] == get_lr(step)
        assert "loss/train" in metrics
        

def test_eval_model(model, rating, text):
    model, loss, confusion_matrix = eval_model(model, rating, text)
    assert model.training == False
    assert isinstance(model, SentimentAnalysisModel)
    assert isinstance(loss, float)
    assert isinstance(confusion_matrix, np.ndarray)
