
import base64
import subprocess
import time
import mlflow
from botocore.exceptions import ClientError
from train.config import TrainingLoopFactory, MLFlowSettings, AWSFactory
from model import ModelFactory
from core.constants import AWS_EC2_STATUS_CODE_RUNNING, MAX_ITERATIONS
from core.logger import get_logger

logger = get_logger(__name__)


MODEL_VARIANTS = {
    "nanoGPT": ModelFactory(
        words=50,
        coordinates=100,
        number_of_blocks=3,
        number_of_heads=5,
        bias = False,
        attention="scaled_dot_product"
    ),
}

# Load the settings from the environment variables and create the AWS clients

aws_factory = AWSFactory()

mlflow_settings = MLFlowSettings(
    is_local=False,
    experiment_id=5
)

model_factory = ModelFactory(
    words=1000,
    coordinates=300,
    number_of_blocks=3,
    number_of_heads=5,
    bias = False,
    attention="metric"
)

training_loop_factory = TrainingLoopFactory()



exports = {
    **model_factory.to_exports(),
    **training_loop_factory.to_exports(),
    **mlflow_settings.to_exports(),
    **aws_factory.to_exports(),

    # other exports

    "COMMIT": subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('ascii'),
}

user_data_exports = ""

for key, value in exports.items():
    user_data_exports += f"export {key}={value}\n"

with open("train/static/user_data.sh", "r") as f:
    user_data_body = f.read()

def create_user_data(run_id: str) -> str:
    user_data = user_data_exports + user_data_body
    user_data = "#!/bin/bash\n" + user_data_exports + f"export MLFLOW_RUN_ID={run_id}\n" + user_data_body
    print(user_data)
    user_data = base64.b64encode(user_data.encode()).decode()
    return user_data


# Create the MLFlow run and start the training loop with the user data

with mlflow.start_run(experiment_id=mlflow_settings.experiment_id) as run:


    mlflow.log_param("commit", exports["COMMIT"])

    ec2_client, cw_client, ec2_resources = aws_factory.create_clients()
    launch_specification = aws_factory.create_launch_specification()
    launch_specification['UserData'] = create_user_data(run.info.run_id)

    for _ in range(MAX_ITERATIONS):

        logger.info(f"Creating request and waiting for fulfilment.")

        spot_request_id = ec2_client.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification=launch_specification
        )['SpotInstanceRequests'][0]['SpotInstanceRequestId']
        ec2_client.get_waiter('spot_instance_request_fulfilled').wait(SpotInstanceRequestIds=[spot_request_id])


        logger.info(f"Spot request {spot_request_id} fulfilled. Waiting for instance creation.")
        instance_id = ec2_client.describe_spot_instance_requests(
            SpotInstanceRequestIds=[spot_request_id]
        )['SpotInstanceRequests'][0]['InstanceId']
        ec2_resources.Instance(instance_id).wait_until_running()

        try:

            logs_history = set()

            while ec2_resources.Instance(instance_id).state['Code'] == AWS_EC2_STATUS_CODE_RUNNING:
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
                    time.sleep(4)

        finally:
            ec2_client.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
            ec2_client.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"Instance {instance_id} terminated.")

        if mlflow.get_run(run.info.run_id).info.status == "FINISHED":
            logger.info("MLFlow run is finished")
            break

        if mlflow.get_run(run.info.run_id).info.status == "FAILED":
            logger.info("MLFlow run has failed")
            break

