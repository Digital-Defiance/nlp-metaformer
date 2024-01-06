
from pydantic_settings import BaseSettings
from typing import  Optional
from core.mixins import MyBaseSettingsMixin
import boto3

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


class TrainConfiguration(BaseSettings, MyBaseSettingsMixin):
    number_of_epochs: int = 100
    number_of_batches: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    loss_function: str = "CrossEntropyLoss"

    class Config:
        env_prefix = "TRAIN_"


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
