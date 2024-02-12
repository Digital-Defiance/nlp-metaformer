
from pyspark.sql import SparkSession
from functools import cache
from pydantic_settings import BaseSettings
import numpy as np
from data.worker import celery_app
import redis


class SparkSettings(BaseSettings):
    driver_memory: str = "2g"
    executor_memory: str = "2g"

    class Config:
        env_prefix = "SPARK_"

    
spark_settings = SparkSettings()

class TrainSettings(BaseSettings):
    n_slices: int = 60

    class Config:
        env_prefix = "TRAIN_"


train_settings = TrainSettings()


@celery_app.task(name='cleanup_task')
def cleanup_task(task_id):
    try:
        task = cleanup_task.AsyncResult(task_id)
        task.forget()
        print(f"Successfully cleaned up task {task_id}")
    except Exception as e:
        print(f"Failed to clean up task {task_id}: {e}")



@celery_app.task(name='prepare_data')
def prepare_data(idx: int, context_window_size: int, seed: int):
    print(f"Slice of index {idx} has been requested.")

    
    print("Starting spark session")
    spark = SparkSession.builder \
                .master("local[*]") \
                .appName("DATA_WORKER") \
                .config("spark.driver.memory", spark_settings.driver_memory) \
                .config("spark.executor.memory", spark_settings.executor_memory) \
                .getOrCreate()

    print("Started spark session")

    train_slices = spark.read.parquet("/data/train.parquet").randomSplit(
        [1.]*train_settings.n_slices,
        seed
    )
    
    print("Generated random split.")

    
    print("Collecting slice...")
    rating, text = [], []
    for row in train_slices[idx].collect():
        rating.append(row.rating)
        text.append(row.text)
    print("Slice has been collected.")
    text_bw = np.array(text)[:, :context_window_size]
    rating_b5 = np.array(rating)

    print("Cleaning up spark resources...")
    spark.stop()
    print("Done.")

    # transform to index of along last dimension TODO: should also cut along the largest ctx window to reduce padding
    rating_b = np.argmax(rating_b5 == 1, axis=-1)
    import uuid
    rating_b_path = f"/data/{uuid.uuid4().hex}"
    text_bw_path = f"/data/{uuid.uuid4().hex}"
    numpy.save(rating_b_path, rating_b)
    numpy.save(text_bw_path, text_bw)
    return rating_b_path, text_bw_path




