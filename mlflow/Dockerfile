
FROM ghcr.io/mlflow/mlflow:v2.9.2

RUN apt update && \ 
    apt install -y gcc libpq-dev && \
    pip install psycopg2==2.8.5 && \
    pip install boto3==1.34.8


LABEL traefik.enable "true"
LABEL traefik.http.routers.mlflow.rule "PathPrefix(`/`)"
LABEL traefik.http.services.mlflow.loadbalancer.server.port "80"

ENV MLFLOW_HOST 0.0.0.0
ENV MLFLOW_PORT 80

CMD mlflow server
