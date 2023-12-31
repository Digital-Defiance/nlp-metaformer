
import time
import boto3
from pydantic_settings import BaseSettings
import subprocess
import os
import base64
import logging
from logging import getLogger
from train_config import TrainConfiguration, ModelHandler, MLFlowSettings
import mlflow

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",)

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_ITERATIONS = 5
AWS_EC2_STATUS_CODE_RUNNING = 16

class AWSSettings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str = "eu-west-2"
    ami_id: str = "ami-093cb9fb2d34920ad"
    instance_type: str = "c5n.xlarge"

    def to_launch_specification(self, user_data):
        return {
            'ImageId': self.ami_id,
            'InstanceType': self.instance_type,
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

            ]
        }

    
aws_settings = AWSSettings()
mlflow_settings = MLFlowSettings()
mlflow.set_tracking_uri(mlflow_settings.TRACKING_URL)

with open("user_data.sh", "r") as f:
    bash_script_template = f.read()

with mlflow.start_run(experiment_id=mlflow_settings.experiment_id) as run:
    TrainConfiguration().save_to_mlflow()
    ModelHandler().save_to_mlflow()

    bash_script = bash_script_template.format(
        current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii'),
        TRACKING_URL=mlflow_settings.TRACKING_URL,
        EXPERIMENT_ID=mlflow_settings.EXPERIMENT_ID,
        MLFLOW_TRACKING_USERNAME=mlflow_settings.MLFLOW_TRACKING_USERNAME,
        MLFLOW_TRACKING_PASSWORD=mlflow_settings.MLFLOW_TRACKING_PASSWORD,
        AWS_ACCESS_KEY_ID=aws_settings.aws_access_key_id,
        AWS_SECRET_ACCESS_KEY=aws_settings.aws_secret_access_key,
        RUN_ID=run.info.run_id,
    )

    user_data = base64.b64encode(bash_script.encode()).decode()
    
    aws_spot = boto3.client(
        'ec2',
        aws_access_key_id=aws_settings.aws_access_key_id,
        aws_secret_access_key=aws_settings.aws_secret_access_key,
        region_name=aws_settings.region_name,
    )


    def get_first_spot_req(description):
        return description['SpotInstanceRequests'][0]

    for _ in range(MAX_ITERATIONS):
    
        if mlflow.get_run(run.info.run_id).info.status == "FINISHED":
            logger.info("MLFlow run is finished")
            break

        response = aws_spot.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification=aws_settings.to_launch_specification(user_data)
        )
        spot_request_id = get_first_spot_req(response)['SpotInstanceRequestId']
        logger.info(f"Spot request {spot_request_id} created. Waiting for fulfilment.")

        aws_spot \
            .get_waiter('spot_instance_request_fulfilled') \
            .wait(SpotInstanceRequestIds=[spot_request_id])

        logger.info(f"Spot request {spot_request_id} fulfilled. Waiting for instance creation.")

        response = aws_spot.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
        instance_id = get_first_spot_req(response)['InstanceId']
        logger.info(f"Instance {instance_id} created. Waiting for running.")
        aws_spot.Instance(instance_id).wait_until_running()

        try:
            while aws_spot.Instance(instance_id).state['Code'] == AWS_EC2_STATUS_CODE_RUNNING:
                logger.info("Instance is running")
                time.sleep(10)
        finally:
            aws_spot.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            logger.info(f"Spot request {spot_request_id} cancelled. Waiting for instance termination.")
            aws_spot.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} terminated.")




