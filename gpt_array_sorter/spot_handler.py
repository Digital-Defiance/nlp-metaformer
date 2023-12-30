import boto3
from pydantic_settings import BaseSettings
from contextlib import contextmanager
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

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

class AWSSpot(BaseSettings):

    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str = "eu-west-2"
    ami_id: str = "ami-09477e8d34e7ae7a2"
    instance_type: str = "t2.micro"
    max_price: str = "0.1"

    @classmethod
    @contextmanager
    def make_spot(cls, mlflow_handler: MLFlowHandler):
        self = cls()

        logger.info("Creating spot instance request")

        self._handler = boto3.client(
            'ec2',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )
    
        current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii')

        MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
        MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")


        logger.info(f"Requesting spot instance for commit {current_commit}")
        response = self._handler.request_spot_instances(
            InstanceCount=1,
            LaunchSpecification={
                'ImageId': self.ami_id,
                'InstanceType': self.instance_type,
                'UserData': base64.b64encode(f"""
                    #!/bin/bash

                    export TRACKING_URL={mlflow_handler.TRACKING_URL}
                    export EXPERIMENT_ID={mlflow_handler.EXPERIMENT_ID}
        
                    export MLFLOW_TRACKING_USERNAME={MLFLOW_TRACKING_USERNAME}
                    export MLFLOW_TRACKING_PASSWORD={MLFLOW_TRACKING_PASSWORD}

                    export AWS_ACCESS_KEY_ID={self.aws_access_key_id}
                    export AWS_SECRET_ACCESS_KEY={self.aws_secret_access_key}

                    # Install dependencies
                    sudo apt-get update
                    git clone github.com/digital-defiance/llm-voice-chat.git
                    cd llm-voice-chat
                    git checkout {current_commit}
                    pip install -r devcontainer/requirements.txt
                    cd gpt_array_sorter
                    python train_worker.py

                """.encode()).decode(),
            },
            SpotPrice=self.max_price,
            Type='one-time',
        )

        spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        logger.info(f"Spot request {spot_request_id} created")
        self._handler.get_waiter('spot_instance_request_fulfilled').wait(SpotInstanceRequestIds=[spot_request_id])
        logger.info(f"Spot request {spot_request_id} fulfilled")

        response = self._handler.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
        instance_id = response['SpotInstanceRequests'][0]['InstanceId']
        ec2_resource = boto3.resource(
            'ec2',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )
        ec2_resource.Instance(instance_id).wait_until_running()
        logger.info(f"Instance {instance_id} is running")

        try:
            yield self
        finally:
            logger.info(f"Cancelling spot request {spot_request_id}")
            self._handler.cancel_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            logger.info(f"Spot request {spot_request_id} cancelled")
            logger.info(f"Stopping instance {instance_id}")
            self._handler.terminate_instances(
                InstanceIds=[instance_id]
            )
            logger.info(f"Instance {instance_id} stopped")

    def is_active(self):
        response = self._handler.describe_spot_instance_requests()
        spot_request = response['SpotInstanceRequests'][0]
        return spot_request['State'] == 'active'