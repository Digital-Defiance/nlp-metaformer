from prefect import task, flow, get_run_logger # type: ignore


import numpy as np
from functools import cache

from pyspark.sql import SparkSession
from pyspark import Row # type: ignore
from pyspark.sql.functions import size, expr, udf
from pyspark.sql.types import StringType, ShortType, StructType, StructField
from pyspark.sql.functions import col
import os
from typing import Callable
from transformers import AutoTokenizer # type: ignore
from transformers import BatchEncoding # type: ignore
from pyspark.sql import DataFrame
from typing import Protocol
from numpy.typing import NDArray
from typing import Any

class Logger(Protocol):
    def info(self, message: str) -> None:
        ...
    
    def error(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...
    
    def debug(self, message: str) -> None:
        ...

get_run_logger: Callable[[], Logger] 


Unknown = object
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased") # type: ignore
ASA_ZIP_URL = "https://github.com/Digital-Defiance/llm-voice-chat/releases/download/dataset-release-amazon-reviews/original_amazon_review_full_csv.tar.gz"
CSV_SCHEMA = StructType([
    StructField("rating", ShortType(), nullable=False),
    StructField("title", StringType(), nullable=False),
    StructField("text", StringType(), nullable=False)
])

@task
def download_zip_file(download_url: str = ASA_ZIP_URL) -> None:

    if os.path.exists("original_amazon_review_full_csv.tar.gz"):
        return

    import requests # type: ignore
    response = requests.get(download_url, stream=True)
    with open("original_amazon_review_full_csv.tar.gz", "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


@task
def extract_zip_file() -> None:
    from prefect import get_run_logger # type: ignore

    logger: Logger = get_run_logger() # type: ignore

    # check if file already exists

    if os.path.exists("amazon_review_full_csv"):
        logger.info("File already exists.")
        return
    
    import tarfile

    # extract file
    with tarfile.open("original_amazon_review_full_csv.tar.gz", "r:gz") as tar:
        tar.extractall()

    for dirname, _, filenames in os.walk('./amazon_review_full_csv'):
        for filename in filenames:
            logger.info(os.path.join(dirname, filename))


@task
@cache
def get_or_create_spark_session() -> SparkSession:
    return ( # type: ignore
      SparkSession 
        .builder
        .master("local[*]") # type: ignore
        .appName("EDA") # type: ignore
        .config("spark.driver.memory", "10g") # type: ignore
        .config("spark.executor.memory", "10g") # type: ignore 
        .getOrCreate()  # type: ignore
    ) 


@task
@cache
def read_csv(spark_session: SparkSession, path: str) -> DataFrame:
    return (
      spark_session
        .read
        .schema(CSV_SCHEMA)
        .option("header", False)
        .option("mode", "FAILFAST")
        .option("inferSchema", False)
        .option("escape", '"')
        .option("multiLine", True)
        .csv(path, mode="FAILFAST")
  )


from torch import Tensor


def tokenize_text(text: str) -> list[int]:

    output: BatchEncoding = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )

    output_tensor: Tensor = output["input_ids"] # type: ignore
    output_list: list[list[int]] = output_tensor.tolist() # type: ignore
    return output_list[0]

from pyspark.sql.types import ArrayType, IntegerType


@task
def calculate_size_of_embedings(data: DataFrame) -> list[Row]:
    append_fields = "case when title is null then text else concat(title, '. ', text) end as text"
    tokenize_result_type = ArrayType(IntegerType())
    spark_tokenize_text = udf(tokenize_text, tokenize_result_type)
    
    return (
      data
        .select(expr(append_fields).alias("text"))
        .select(expr("lower(text) as text"))
        .select(spark_tokenize_text("text").alias("text"))
        .select(size("text").alias("length"))
        .collect()
    )

@task 
def to_numpy(list_of_rows: list[Row]) -> NDArray[Any]:
    return np.array(list_of_rows)

@flow
def finding_ctx_window() -> None:
    logger: Logger = get_run_logger() 

    download_zip_file()
    extract_zip_file()

    spark_session = get_or_create_spark_session()

    for path in [
        # "amazon_review_full_csv/test.csv",
        "amazon_review_full_csv/train.csv",
    ]:
        sizes_path = path.replace(".csv", "_sizes.npy")

        if os.path.exists(sizes_path):
            logger.info(f"File {sizes_path} already exists.")
            continue

        data = read_csv(spark_session, path)
        sizes = calculate_size_of_embedings(data)
        sizes = to_numpy(sizes)
        np.save(sizes_path, sizes)



@task
def preprocess_dataset(data: DataFrame) -> list[Row]:
    append_fields = "case when title is null then text else concat(title, '. ', text) end as text"
    tokenize_result_type = ArrayType(IntegerType())
    spark_tokenize_text = udf(tokenize_text, tokenize_result_type)
    
    return ( # type: ignore
      data  
        .select("rating", expr(append_fields).alias("text"))
        .select("rating", expr("lower(text) as text"))
        .select("rating", spark_tokenize_text("text").alias("text"))
        .select(size("text").alias("size"), "rating", "text")
        .filter(col("size") <= 300)
        .select("rating", expr(f"concat(text, array_repeat(0, 300 - size(text))) as text"))
    )





@task
def read_parquet(spark_session: SparkSession, path: str) -> DataFrame:
    return spark_session.read.parquet(path)


@task
def df_to_numpy(df: DataFrame) -> tuple[Any, Any]:
    rating = df.select("rating").collect()
    text = df.select("text").collect()
    return np.array(rating, dtype=np.int64), np.array(text, dtype=np.int32)


@flow
def parquet_partitions_to_npz() -> None:
    logger: Logger = get_run_logger() 

    # download_zip_file()
    # extract_zip_file()

    spark_session = get_or_create_spark_session()

    for path in [
        # "amazon_review_full_csv/test.csv",
        "amazon_review_full_csv/train.csv",
    ]:

        parquet_path = path.replace(".csv", "/tmp")
        if not os.path.exists(parquet_path):
            data = read_csv(spark_session, path)
            data = preprocess_dataset(data)
            data.repartition(10).write.parquet(parquet_path) # type: ignore


        rating = np.array([0], dtype=np.int64)
        text = np.array([[0] * 300], dtype=np.uint32)
        idx = 0
        for file in os.listdir(parquet_path):
            if file.endswith(".parquet"):
                path = os.path.join(parquet_path, file)
                partition: DataFrame = read_parquet(spark_session, path) # type: ignore
                rating_slice, text_slice = df_to_numpy(partition) # type: ignore
                rating_slice = rating_slice[:, 0]
                text_slice = text_slice[:, 0, :]
                save_path = os.path.join(parquet_path, f"{idx}.npz")
                np.savez_compressed(save_path, rating=rating_slice, text=text_slice)
                # rating = np.append(rating, rating_slice)
                # text = np.append(text, text_slice, axis=0)
                print(rating.shape, text.shape)
                idx += 1
    
        # np.savez_compressed(dataset_path, rating=rating[1:], text=text[1:])


@flow
def join_npz() -> None:

    for split in [
        # "amazon_review_full_csv/test.csv",
        "train",
    ]:
        npz_path = f"amazon_review_full_csv/{split}/tmp"
        

        rating = np.array([0], dtype=np.int64)
        text = np.array([[0] * 300], dtype=np.uint32)
        idx = 0
        for file in os.listdir(npz_path):
            if file.endswith(".npz"):
                path = os.path.join(npz_path, file)
                data = np.load(path)
                rating_slice = data["rating"]
                text_slice = data["text"]
                print(rating_slice.shape, text_slice.shape)
                rating = np.append(rating, rating_slice)
                text = np.append(text, text_slice, axis=0)
                print(rating.shape, text.shape)
                if idx == 5:
                    dataset_path = f"amazon_review_full_csv/{split}_0.npz"
                    np.savez_compressed(dataset_path, rating=rating[1:], text=text[1:])
                    rating = np.array([0], dtype=np.int64)
                    text = np.array([[0] * 300], dtype=np.uint32)


                idx += 1
        dataset_path = f"amazon_review_full_csv/{split}_1.npz"
        np.savez_compressed(dataset_path, rating=rating[1:], text=text[1:])

if __name__ == "__main__":
    # parquet_partitions_to_npz()
    join_npz()

