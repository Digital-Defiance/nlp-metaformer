
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

MAX_ITERATIONS = 5
AWS_EC2_STATUS_CODE_RUNNING = 16

class AWSSpot(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str = "eu-west-2"
    ami_id: str = "ami-093cb9fb2d34920ad"
    instance_type: str = "c5n.xlarge"

    def create_client(self):
        return boto3.client(
            'ec2',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

with MLFlowHandler.start_run() as mlflow_handler:
    TrainConfiguration().save_to_mlflow(mlflow_handler)
    ModelHandler().save_to_mlflow(mlflow_handler)
    spot_parameters = AWSSpot()
    aws_spot = spot_parameters.create_client()

    with open("user_data.sh", "r") as f:
        bash_script_template = f.read()

    bash_script = bash_script_template.format(
        current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii'),
        TRACKING_URL=mlflow_handler.TRACKING_URL,
        EXPERIMENT_ID=mlflow_handler.EXPERIMENT_ID,
        MLFLOW_TRACKING_USERNAME=os.environ.get("MLFLOW_TRACKING_USERNAME"),
        MLFLOW_TRACKING_PASSWORD=os.environ.get("MLFLOW_TRACKING_PASSWORD"),
        AWS_ACCESS_KEY_ID=spot_parameters.aws_access_key_id,
        AWS_SECRET_ACCESS_KEY=spot_parameters.aws_secret_access_key,
        RUN_ID=mlflow_handler._run_id
    )

    user_data = base64.b64encode(bash_script.encode()).decode()

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
                'UserData': user_data,
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
            while aws_spot.Instance(instance_id).state['Code'] == AWS_EC2_STATUS_CODE_RUNNING:
                logger.info("Instance is running")
                time.sleep(10)
        finally:
            aws_spot.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            logger.info(f"Spot request {spot_request_id} cancelled. Waiting for instance termination.")
            aws_spot.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} terminated.")




