
import base64
import logging
import subprocess
import time
from logging import getLogger
import boto3
import mlflow
from botocore.exceptions import ClientError

from pydantic_settings import BaseSettings
from train.config import TrainConfiguration, ModelHandler, MLFlowSettings


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",)

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_ITERATIONS = 3
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

with open("user_data.sh", "r") as f:
    bash_script_template = f.read()


with mlflow.start_run(experiment_id=mlflow_settings.experiment_id) as run:
    TrainConfiguration().save_to_mlflow()
    ModelHandler().save_to_mlflow()

    bash_script = bash_script_template.format(
        current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii'),
        TRACKING_URI=mlflow_settings.tracking_uri,
        EXPERIMENT_ID=mlflow_settings.experiment_id,
        MLFLOW_TRACKING_USERNAME=mlflow_settings.tracking_username,
        MLFLOW_TRACKING_PASSWORD=mlflow_settings.tracking_password,
        AWS_ACCESS_KEY_ID=aws_settings.aws_access_key_id,
        AWS_SECRET_ACCESS_KEY=aws_settings.aws_secret_access_key,
        RUN_ID=run.info.run_id,
    )

    user_data = base64.b64encode(bash_script.encode()).decode()

    boto_kwargs = {
        "aws_access_key_id": aws_settings.aws_access_key_id,
        "aws_secret_access_key": aws_settings.aws_secret_access_key,
        "region_name": aws_settings.region_name,
    }   
    
    ec2_client = boto3.client('ec2', **boto_kwargs)
    cw_client = boto3.client('logs', **boto_kwargs)
    ec2_resources = boto3.resource('ec2', **boto_kwargs)


    for _ in range(MAX_ITERATIONS):

        response = ec2_client.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification=aws_settings.to_launch_specification(user_data)
        )
        spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        logger.info(f"Spot request {spot_request_id} created. Waiting for fulfilment.")

        spot_request_waiter = ec2_client.get_waiter('spot_instance_request_fulfilled')
        spot_request_waiter.wait(SpotInstanceRequestIds=[spot_request_id])

        logger.info(f"Spot request {spot_request_id} fulfilled. Waiting for instance creation.")

        response = ec2_client.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
        instance_id = response['SpotInstanceRequests'][0]['InstanceId']
        logger.info(f"Instance {instance_id} created. Waiting for running.")

        def fetch_instance():
            return ec2_resources.Instance(instance_id)

        fetch_instance().wait_until_running()
        logger.info("Instance is running")

        try:

            logs_history = set()

            while fetch_instance().state['Code'] == AWS_EC2_STATUS_CODE_RUNNING:
                time.sleep(1)

                # Get the logs from CloudWatch

                try:
                    response = cw_client.get_log_events(
                        logGroupName="/var/log/cloud-init-output.log",
                        logStreamName=instance_id,
                    )

                    for event in response['events']:
                        if event['message'] in logs_history:
                            continue

                        logs_history.add(event['message'])
                        print(f"{instance_id}> {event['message']}")
        
                except ClientError:
                    logger.info("Log group not found yet")

        finally:
            ec2_client.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            logger.info(f"Spot request {spot_request_id} cancelled. Waiting for instance termination.")
            ec2_client.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} terminated.")

        if mlflow.get_run(run.info.run_id).info.status == "FINISHED":
            logger.info("MLFlow run is finished")
            break

        if mlflow.get_run(run.info.run_id).info.status == "FAILED":
            logger.info("MLFlow run has failed")
            break


