
from spot_handler import AWSSpot
from mlflow_handler import MLFlowHandler
from model_handler import ModelHandler
import mlflow
import time
import boto3
from pydantic_settings import BaseSettings
import subprocess
from mlflow_handler import MLFlowHandler
import os
import base64
import logging
from logging import getLogger

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = getLogger(__name__)
logger.setLevel(logging.INFO)


class AWSSpot(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str = "eu-west-2"
    ami_id: str = "ami-093cb9fb2d34920ad"
    instance_type: str = "c5n.xlarge"
    max_price: str = "5"

MAX_ITERATIONS = 5
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")
spot_parameters = AWSSpot()
model_parameters = ModelHandler.export_parameters()
current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii')


with MLFlowHandler.start_run() as mlflow_handler:
    mlflow.log_param("status", "STARTING")
    mlflow.log_param("model_definition", model_parameters)

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"Starting iteration {iteration}")

        if not mlflow_handler.is_active():
            logger.info("MLFlow run is not active")
            break

        logger.info("Creating spot instance request")
        aws_spot = boto3.client(
            'ec2',
            aws_access_key_id=spot_parameters.aws_access_key_id,
            aws_secret_access_key=spot_parameters.aws_secret_access_key,
            region_name=spot_parameters.region_name
        )

        logger.info(f"Requesting spot instance for commit {current_commit}")
        response = aws_spot.request_spot_instances(
            InstanceCount=1,
            LaunchSpecification={
                'ImageId': spot_parameters.ami_id,
                'InstanceType': spot_parameters.instance_type,
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

                ],
                'UserData': base64.b64encode(f"""#!/bin/bash
                                             

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

# Add the code to append to the bash profile
echo 'export TRACKING_URL={mlflow_handler.TRACKING_URL}' >> ~/.bash_profile
echo 'export EXPERIMENT_ID={mlflow_handler.EXPERIMENT_ID}' >> ~/.bash_profile
echo 'export MLFLOW_TRACKING_USERNAME={MLFLOW_TRACKING_USERNAME}' >> ~/.bash_profile
echo 'export MLFLOW_TRACKING_PASSWORD={MLFLOW_TRACKING_PASSWORD}' >> ~/.bash_profile
echo 'export AWS_ACCESS_KEY_ID={spot_parameters.aws_access_key_id}' >> ~/.bash_profile
echo 'export AWS_SECRET_ACCESS_KEY={spot_parameters.aws_secret_access_key}' >> ~/.bash_profile

python -m venv env
source env/bin/activate
pip install -r .devcontainer/requirements.txt
cd gpt_array_sorter
python train_worker.py
# shutdown -h now

                """.encode()).decode(),
            },
            Type='one-time',
        )
        spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        logger.info(f"Spot request {spot_request_id} created")
        aws_spot.get_waiter('spot_instance_request_fulfilled').wait(SpotInstanceRequestIds=[spot_request_id])


        logger.info(f"Spot request {spot_request_id} fulfilled")
        response = aws_spot.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
        instance_id = response['SpotInstanceRequests'][0]['InstanceId']

        spot_instance = boto3.resource(
            'ec2',
            aws_access_key_id=spot_parameters.aws_access_key_id,
            aws_secret_access_key=spot_parameters.aws_secret_access_key,
            region_name=spot_parameters.region_name
        ).Instance(instance_id).wait_until_running()


        logger.info(f"Instance {instance_id} is running")

        try:
            while boto3.resource(
                'ec2',
                aws_access_key_id=spot_parameters.aws_access_key_id,
                aws_secret_access_key=spot_parameters.aws_secret_access_key,
                region_name=spot_parameters.region_name
            ).Instance(instance_id).state['Code'] == 16:
                logger.info("Instance is running")
                time.sleep(10)
        finally:
            logger.info(f"Cancelling spot request {spot_request_id}")
            aws_spot.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            logger.info(f"Spot request {spot_request_id} cancelled")
            logger.info(f"Stopping instance {instance_id}")
            aws_spot.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} stopped")




