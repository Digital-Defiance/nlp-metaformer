FROM python:3.11.7-bullseye


RUN apt update && apt install -y openjdk-11-jre-headless && pip install celery==5.3.6 redis==5.0.0 pyspark==3.5.0 pydantic-settings==2.1.0 numpy==1.26.4

ENV C_FORCE_ROOT="true"

RUN mkdir /app
WORKDIR /app
COPY main.py /app/main.py
COPY worker.py /app/worker.py
RUN celery -A main worker
