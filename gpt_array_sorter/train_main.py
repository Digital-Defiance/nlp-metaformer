
import time
import boto3
from pydantic_settings import BaseSettings
import subprocess
import os
import base64
import logging
from logging import getLogger
from train_config import TrainConfiguration, ModelHandler, MLFlowHandler

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",)

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

class AWSSpot(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str = "eu-west-2"
    ami_id: str = "ami-093cb9fb2d34920ad"
    instance_type: str = "c5n.xlarge"
    max_price: str = "5"

    def create_client(self):
        return boto3.client(
            'ec2',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )


MAX_ITERATIONS = 5

def make_user_data(mlflow_handler, spot_parameters):
    MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii')
    return base64.b64encode(f"""#!/bin/bash
sudo mkdir /larger_tmp
export TMPDIR=/larger_tmp

sudo fallocate -l 30G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile1
sudo swapon /swapfile

sudo yum update -y 
sudo yum install -y git  
sudo yum install -y python
sudo yum install -y python3-pip

git clone https://github.com/Digital-Defiance/llm-voice-chat.git
cd llm-voice-chat
git checkout {current_commit}

export TRACKING_URL={mlflow_handler.TRACKING_URL}
export EXPERIMENT_ID={mlflow_handler.EXPERIMENT_ID}
export MLFLOW_TRACKING_USERNAME={MLFLOW_TRACKING_USERNAME}
export MLFLOW_TRACKING_PASSWORD={MLFLOW_TRACKING_PASSWORD}
export AWS_ACCESS_KEY_ID={spot_parameters.aws_access_key_id}
export AWS_SECRET_ACCESS_KEY={spot_parameters.aws_secret_access_key}
export RUN_ID={mlflow_handler._run_id}


python -m venv env
source env/bin/activate
pip install -r .devcontainer/requirements.txt
cd gpt_array_sorter
python train_worker.py
shutdown -h now
                """.encode()).decode()


with MLFlowHandler.start_run() as mlflow_handler:
    TrainConfiguration().save_to_mlflow(mlflow_handler)
    ModelHandler().save_to_mlflow(mlflow_handler)
    spot_parameters = AWSSpot()
    aws_spot = spot_parameters.create_client()
    for _ in range(MAX_ITERATIONS):
        if mlflow_handler.get_status() == "FINISHED":
            logger.info("MLFlow run is finished")
            break

        spot_request_id = aws_spot.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification={
                'ImageId': spot_parameters.ami_id,
                'InstanceType': spot_parameters.instance_type,
                'KeyName': 'r',
                'UserData': make_user_data(mlflow_handler, spot_parameters),
                'BlockDeviceMappings': [
                    {
                        'DeviceName': '/dev/xvda',
                        'Ebs': {
                            'VolumeSize': 130,  
                            'DeleteOnTermination': True,
                            'VolumeType': 'gp2',
                        },
                    },

                ],
                
            },
        )['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        logger.info(f"Spot request {spot_request_id} created. Waiting for fulfilment.")

        aws_spot \
            .get_waiter('spot_instance_request_fulfilled') \
            .wait(SpotInstanceRequestIds=[spot_request_id])
        logger.info(f"Spot request {spot_request_id} fulfilled. Waiting for instance creation.")

        instance_id = aws_spot.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])['SpotInstanceRequests'][0]['InstanceId']
        logger.info(f"Instance {instance_id} created. Waiting for running.")

        aws_spot.Instance(instance_id).wait_until_running()
        logger.info(f"Instance {instance_id} is running. Waiting for instance status check.")

        try:
            AWS_EC2_STATUS_CODE_RUNNING = 16
            while aws_spot.Instance(instance_id).state['Code'] == AWS_EC2_STATUS_CODE_RUNNING:
                logger.info("Instance is running")
                time.sleep(10)
        finally:
            logger.info(f"Cancelling spot request {spot_request_id}")
            aws_spot.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            logger.info(f"Spot request {spot_request_id} cancelled")

            logger.info(f"Stopping instance {instance_id}")
            aws_spot.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} stopped")




