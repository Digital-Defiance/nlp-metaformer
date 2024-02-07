
from pydantic_settings import BaseSettings
from typing import  Optional
from core.mixins import MyBaseSettingsMixin
from core.constants import DEVICE
import boto3
import torch
import tiktoken
import numpy as np
import torch.nn as nn
import mlflow
import mlflow.pytorch
from pydantic import FilePath
from pydantic import BeforeValidator
from typing import Literal


gpt2_encoder = tiktoken.get_encoding("gpt2")

def text_to_tensor(text: str) -> torch.Tensor:
    tokens = gpt2_encoder.encode_ordinary(text)
    tokens = np.array(tokens, dtype=np.int32)
    return torch.from_numpy(tokens).to(DEVICE)

TokenizedTextData = torch.Tensor # , BeforeValidator(text_to_tensor)]

class DataFactory(BaseSettings, MyBaseSettingsMixin):
    validation_tokens: TokenizedTextData
    training_tokens: TokenizedTextData
    batch_size: int

    def create_batch(self, split = "train", max_size_of_sequence: int = 1000):
        data_s = self.training_tokens if split == "train" else self.validation_tokens
        shape = (self.batch_size,)
        start_indices_b = torch.randint(0, len(data_s) - max_size_of_sequence, shape).to(DEVICE)
        end_indices_b = start_indices_b + max_size_of_sequence
        shape = (self.batch_size, max_size_of_sequence)
        in_sequence_bw = torch.zeros(shape, dtype=torch.int64, device=DEVICE)
        out_sequence_bw = torch.zeros(shape, dtype=torch.int64, device=DEVICE)

        for i in range(self.batch_size):
            start, end = start_indices_b[i], end_indices_b[i]
            in_sequence_bw[i] = data_s[start:end]
            out_sequence_bw[i] = data_s[start + 1:end + 1]

        return in_sequence_bw, out_sequence_bw




class Adam(torch.optim.Adam):
    def set_lr(self, lr: float):
        for param_group in self.param_groups:
            param_group['lr'] = lr


class TrainingLoopFactory(BaseSettings, MyBaseSettingsMixin):
    number_of_epochs: int = 100
    number_of_batches: int = 10
    loss_function: str = "CrossEntropyLoss"
    batch_size: int = 32
    input_text_file: FilePath = "train/static/raw_data.txt"
    split_ratio: float = 0.9
    l1_regularization: float = 0
    l2_regularization: float = 0

    beta_1: float = 0.9
    beta_2: float = 0.98
    epsilon: float = 1e-9
    warmup_steps: int = 4000

    class Config:
        env_prefix = "TRAIN_"

    def create_data_factory(self) -> DataFactory:

        with self.input_text_file.open() as file:
            text: str = file.read()

        split_idx = int(len(text) * self.split_ratio)

        return DataFactory(
            training_tokens=text[:split_idx],
            validation_tokens=text[split_idx:],
            batch_size=self.batch_size,
        )

    def create_loss_function(self):
        if self.loss_function == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()

    def create_optimizer(self, parameters):
        return Adam(
            parameters,
            lr=1,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon
        )

    def create_scheduler(self, d_model: int):
        return lambda step: min(step ** -0.5, step * self.warmup_steps ** -1.5) * d_model ** -0.5


class MLFlowSettings(BaseSettings, MyBaseSettingsMixin):
    experiment_id: int = 5
    run_id: Optional[str] = None
    tracking_uri: Literal["http://mlflow:80"] 
    tracking_username: str
    tracking_password: str
    log_system_metrics: bool = True
    is_local: bool = True

    class Config:
        env_prefix = "MLFLOW_"


    def has_checkpoint(self) -> bool:
        last_epoch: float | None = mlflow.get_run(self.run_id).data.metrics.get('epoch', None)
        return last_epoch is not None
    
    def load_model(self):
        last_epoch: float | None = mlflow.get_run(self.run_id).data.metrics.get('epoch', None)
        if last_epoch is None:
            raise RuntimeError("No checkpoint found")
        last_epoch = int(last_epoch)
        model_uri = f"runs:/{self.run_id}/nanogpt_{last_epoch}"
        model = mlflow.pytorch.load_model(model_uri)
        return model.to(DEVICE), last_epoch


class AWSFactory(BaseSettings, MyBaseSettingsMixin):
    access_key_id: str
    secret_access_key: str
    default_region_name: str = "eu-west-2"
    user_data: FilePath = "train/static/user_data.sh"
    ami_id: str = "ami-0338002ad7a4a92e1"
    instance_type: str = "c5n.xlarge"

    class Config:
        env_prefix = "AWS_"

    def create_launch_specification(self):
        return {
            'ImageId': self.ami_id,
            'InstanceType': self.instance_type,
            'KeyName': 'r',
            'BlockDeviceMappings': [
                {
                    'DeviceName': '/dev/xvda',
                    'Ebs': {
                        'VolumeSize': 130,  
                        'DeleteOnTermination': True,
                        'VolumeType': 'gp2',
                    },
                },

            ]
        }

    def create_clients(self) -> tuple:
        boto_kwargs = {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "region_name": self.default_region_name,
        }

        ec2_client = boto3.client('ec2', **boto_kwargs)
        cw_client = boto3.client('logs', **boto_kwargs)
        ec2_resources = boto3.resource('ec2', **boto_kwargs)
        return ec2_client, cw_client, ec2_resources
