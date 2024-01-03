#!/bin/bash


export AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}


cat << 'EOF' > send_logs_to_cloudwatch.sh
#!/bin/bash

LOG_FILE=/var/log/cloud-init-output.log
PRIVATE_IP=$(hostname -I)
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=private-ip-address,Values=$PRIVATE_IP" --query "Reservations[*].Instances[*].InstanceId" --output text)
aws logs create-log-stream --log-group-name $LOG_FILE --log-stream-name $INSTANCE_ID

NEXT_SEQUENCE_TOKEN=$(aws logs describe-log-streams --log-group-name $LOG_FILE --log-stream-name $INSTANCE_ID | jq -r '.logStreams[0].uploadSequenceToken')

tail -F $LOG_FILE | while IFS= read -r LINE; do
  if [[ -z "$LINE" ]]; then
    continue
  fi

  TIMESTAMP=$(date -u +%s%3N)  # Current time in milliseconds
  LINE_ESCAPED=$(echo "$LINE" | sed 's/"/\\"/g' | tr -d '\n')  # Escape double quotes and remove newline
  # echo $LINE_ESCAPED
  RESPONSE=$(aws logs put-log-events \
    --log-group-name $LOG_FILE \
    --log-stream-name $INSTANCE_ID \
    --log-events timestamp=$TIMESTAMP,message="'$LINE_ESCAPED'" \
    --sequence-token $NEXT_SEQUENCE_TOKEN)

  if [[ $? -ne 0 ]]; then
    echo "Error sending log event. Response was: $RESPONSE"
    # Handle error, possibly retrieve new sequence token
  else
    NEXT_SEQUENCE_TOKEN=$(echo $RESPONSE | jq -r '.nextSequenceToken')
  fi
done
EOF


chmod +x send_logs_to_cloudwatch.sh
nohup ./send_logs_to_cloudwatch.sh &



export MLFLOW_RUN_ID={RUN_ID}
export MLFLOW_TRACKING_URI={TRACKING_URI}
export MLFLOW_EXPERIMENT_ID={EXPERIMENT_ID}
export MLFLOW_TRACKING_USERNAME={MLFLOW_TRACKING_USERNAME}
export MLFLOW_TRACKING_PASSWORD={MLFLOW_TRACKING_PASSWORD}

sudo mkdir /larger_tmp
export TMPDIR=/larger_tmp

# sudo fallocate -l 30G /swapfile
# sudo chmod 600 /swapfile
# sudo mkswap /swapfile
# sudo swapon /swapfile

sudo yum update -y 
sudo yum install -y git 
sudo yum install -y python
sudo yum install -y python3-pip

git clone https://github.com/Digital-Defiance/llm-voice-chat.git
cd llm-voice-chat
git checkout {current_commit}

python -m venv env
source env/bin/activate
pip install -r .devcontainer/requirements.txt
cd gpt_shakespear
python train_worker.py
#wait two minutes before shutting down, so that the logs can be sent to cloudwatch
shutdown -h +1

