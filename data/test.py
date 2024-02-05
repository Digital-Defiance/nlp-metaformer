from celery_app import celery_app



celery_app.send_task('prepare_data', args=[10], kwargs={})



# add.delay(1)
# apt update
# apt install openjdk-11-jdk -y
# curl -L https://github.com/Digital-Defiance/llm-voice-chat/releases/download/asa-v0.2.0/test.parquet -o test.parquet
# curl -L https://github.com/Digital-Defiance/llm-voice-chat/releases/download/asa-v0.2.0/train.parquet -o train.parquet