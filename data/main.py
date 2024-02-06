
from pyspark.sql import SparkSession
from functools import cache
from pydantic_settings import BaseSettings
import numpy as np
from worker import Worker


class SparkSettings(BaseSettings):
    driver_memory: str = "2g"
    executor_memory: str = "2g"

    class Config:
        env_prefix = "SPARK_"

    
    @classmethod
    @cache
    def get_spark(cls):
        self = cls()
        return SparkSession.builder \
            .master("local[*]") \
            .appName("DATA_WORKER") \
            .config("spark.driver.memory", self.driver_memory) \
            .config("spark.executor.memory", self.executor_memory) \
            .getOrCreate()

    @classmethod
    def get_spark_context(cls):
        return cls.get_spark().sparkContext
    
spark = SparkSettings.get_spark()

class TrainSettings(BaseSettings):
    n_slices: int = 60

    class Config:
        env_prefix = "TRAIN_"

    @classmethod
    def get_train_slices(cls):
        self = cls()
        return spark.read.parquet("train.parquet").randomSplit([1.]*self.n_slices)
    
    @classmethod
    def get_test_slices(cls):
        self = cls()
        return spark.read.parquet("test.parquet").randomSplit([1.]*self.n_slices)


train_slices = TrainSettings.get_train_slices()

main = Worker().celery

@main.task(name='prepare_data')
def prepare_data(idx: int) -> str:
    print(f"GOT INDEX: {idx}")
    rating = []
    text = []
    for row in train_slices[idx].collect():
        rating.append(row.rating)
        text.append(row.text)
    rating  = np.array(rating)
    text = np.array(text)
    return rating, text



