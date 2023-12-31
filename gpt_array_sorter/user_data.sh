#!/bin/bash
sudo mkdir /larger_tmp
export TMPDIR=/larger_tmp

sudo fallocate -l 30G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile1
sudo swapon /swapfile

sudo yum update -y 
sudo yum install -y git  
sudo yum install -y python
sudo yum install -y python3-pip

git clone https://github.com/Digital-Defiance/llm-voice-chat.git
cd llm-voice-chat
git checkout {current_commit}

export MLFLOW_TRACKING_URL={TRACKING_URL}
export MLFLOW_EXPERIMENT_ID={EXPERIMENT_ID}
export MLFLOW_TRACKING_USERNAME={MLFLOW_TRACKING_USERNAME}
export MLFLOW_TRACKING_PASSWORD={MLFLOW_TRACKING_PASSWORD}
export AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
export MLFLOW_RUN_ID={RUN_ID}

python -m venv env
source env/bin/activate
pip install -r .devcontainer/requirements.txt
cd gpt_array_sorter
python train_worker.py
# shutdown -h now