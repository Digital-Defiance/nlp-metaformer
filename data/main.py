
import redis
from pyspark.sql import SparkSession
from celery_app import celery_app



spark = SparkSession.builder \
    .master("local[*]") \
    .appName("DATA_WORKER") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()


test = spark.read.parquet("test.parquet")
train = spark.read.parquet("train.parquet") 

redis_db1 = redis.Redis(host='redis', port=6379, db=1, decode_responses=True)

@celery_app.task(name='prepare_data')
def prepare_data(idx: int) -> str:
    print(idx)
    test.show()
    return "Success!"
    

