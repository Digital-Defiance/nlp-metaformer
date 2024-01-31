FROM python:3.8

RUN apt update && apt install docker.io -y \
               && apt install awscli -y


RUN pip install torch==2.1.2 boto3==1.34.8 psutil==5.9.7 mlflow==2.9.2 pydantic==2.5.3 pydantic-settings==2.1.0 tqdm==4.66.1 tiktoken==0.5.2 pynvml
