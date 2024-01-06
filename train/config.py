
from pydantic_settings import BaseSettings
from typing import  Optional
from core.mixins import MyBaseSettingsMixin
from core.constants import DEVICE
import boto3
from typing import Iterator
import torch
import tiktoken
import numpy as np
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch


gpt2_encoder = tiktoken.get_encoding("gpt2")
input_file_path = "raw_data.txt"

with open(input_file_path, 'r') as file:
    text: str = file.read()

size_of_text = len(text)
thresh_size = int(size_of_text * 0.9)
train_data, val_data = text[:thresh_size], text[thresh_size:]

train_ids = gpt2_encoder.encode_ordinary(train_data)
train_ids = np.array(train_ids, dtype=np.int32)
train_ids = torch.from_numpy(train_ids).to(DEVICE)

val_ids = gpt2_encoder.encode_ordinary(val_data)
val_ids = np.array(val_ids, dtype=np.int32)
val_ids = torch.from_numpy(val_ids).to(DEVICE)



class TrainingLoopFactory(BaseSettings, MyBaseSettingsMixin):
    number_of_epochs: int = 100
    number_of_batches: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    loss_function: str = "CrossEntropyLoss"

    class Config:
        env_prefix = "TRAIN_"

    def _create_batch(self, data_s: torch.Tensor, max_size_of_sequence: int):

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
    
    def create_validation_batch(self):
        return self._create_batch(val_ids, 1000)

    def create_training_batch(self):
        return self._create_batch(train_ids, 1000)

    def create_epoch_data(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(self.number_of_batches):
            yield self.generate_batch(train_ids)

    def create_loss_function(self):
        if self.loss_function == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        raise RuntimeError(f"Unknown loss function {self.loss_function}")
    
    def create_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.learning_rate)



class MLFlowSettings(BaseSettings, MyBaseSettingsMixin):
    experiment_id: int
    run_id: Optional[str] = None
    tracking_uri: str
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
    ami_id: str = "ami-093cb9fb2d34920ad"
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
            "default_region_name": self.region_name,
        }

        ec2_client = boto3.client('ec2', **boto_kwargs)
        cw_client = boto3.client('logs', **boto_kwargs)
        ec2_resources = boto3.resource('ec2', **boto_kwargs)
        return ec2_client, cw_client, ec2_resources
