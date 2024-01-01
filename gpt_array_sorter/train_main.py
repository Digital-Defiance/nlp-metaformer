
import time
import boto3
from pydantic_settings import BaseSettings
import subprocess
import base64
import logging
from logging import getLogger
from train_config import TrainConfiguration, ModelHandler, MLFlowSettings
import mlflow
from botocore.exceptions import ClientError



logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",)

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_ITERATIONS = 1
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
mlflow.set_tracking_uri(mlflow_settings.tracking_url)

with open("user_data.sh", "r") as f:
    bash_script_template = f.read()

with mlflow.start_run(experiment_id=mlflow_settings.experiment_id) as run:
    TrainConfiguration().save_to_mlflow()
    ModelHandler().save_to_mlflow()

    bash_script = bash_script_template.format(
        current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii'),
        TRACKING_URL=mlflow_settings.tracking_url,
        EXPERIMENT_ID=mlflow_settings.experiment_id,
        MLFLOW_TRACKING_USERNAME=mlflow_settings.tracking_username,
        MLFLOW_TRACKING_PASSWORD=mlflow_settings.tracking_password,
        AWS_ACCESS_KEY_ID=aws_settings.aws_access_key_id,
        AWS_SECRET_ACCESS_KEY=aws_settings.aws_secret_access_key,
        RUN_ID=run.info.run_id,
    )


    user_data = base64.b64encode(bash_script.encode()).decode()
    
    ec2_client = boto3.client(
        'ec2',
        aws_access_key_id=aws_settings.aws_access_key_id,
        aws_secret_access_key=aws_settings.aws_secret_access_key,
        region_name=aws_settings.region_name,
    )

    ec2_resources = boto3.resource(
        'ec2',
        aws_access_key_id=aws_settings.aws_access_key_id,
        aws_secret_access_key=aws_settings.aws_secret_access_key,
        region_name=aws_settings.region_name,
    )

    cw_client = client = boto3.client(
        'logs',
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

        response = ec2_client.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification=aws_settings.to_launch_specification(user_data)
        )
        spot_request_id = get_first_spot_req(response)['SpotInstanceRequestId']
        logger.info(f"Spot request {spot_request_id} created. Waiting for fulfilment.")

        ec2_client \
            .get_waiter('spot_instance_request_fulfilled') \
            .wait(SpotInstanceRequestIds=[spot_request_id])

        logger.info(f"Spot request {spot_request_id} fulfilled. Waiting for instance creation.")

        response = ec2_client.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
        instance_id = get_first_spot_req(response)['InstanceId']
        logger.info(f"Instance {instance_id} created. Waiting for running.")
        ec2_resources.Instance(instance_id).wait_until_running()
        logger.info("Instance is running")

        try:

            logs = set()
            while ec2_resources.Instance(instance_id).state['Code'] == AWS_EC2_STATUS_CODE_RUNNING:
                time.sleep(1)

                # Get the logs from CloudWatch

                try:
                    response = cw_client.get_log_events(
                        logGroupName="/var/log/cloud-init-output.log",
                        logStreamName=instance_id,
                    )

                    for event in response['events']:
                        if event['message'] in logs:
                            continue

                        logs.add(event['message'])
                        logger.info(f"{instance_id}> {event['message']}")
        
                except ClientError:
                    logger.info("Log group not found yet")

        finally:
            ec2_client.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            logger.info(f"Spot request {spot_request_id} cancelled. Waiting for instance termination.")
            ec2_client.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} terminated.")




