FROM python:3.8

RUN apt update && apt install docker.io -y \
               && apt install awscli -y

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
