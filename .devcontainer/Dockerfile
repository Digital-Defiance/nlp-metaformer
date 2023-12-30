FROM spotty-instance-gpt-array-sorter-20-aws-1-BuildingDockerImageSignal-114L9OI9A07Z9

RUN apt update && apt install docker.io -y \
               && apt install awscli -y

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt