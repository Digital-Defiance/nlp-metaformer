import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType
import tiktoken
import tiktoken
import numpy as np
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import size, max
from pyspark.sql.functions import col, expr

class Paths:
    TRAIN_DATA = "./amazon_review_full_csv/train.csv"
    TEST_DATA = "./amazon_review_full_csv/test.csv"


class SentimentAnalysisTokenizer:
    def __init__(self):
        p50k_base = tiktoken.get_encoding("p50k_base")

        p50k_no_special_tokens = tiktoken.Encoding(

            name="p50k_base_no_special_tokens",
            pat_str=p50k_base._pat_str,
            mergeable_ranks=p50k_base._mergeable_ranks,
            special_tokens={}
        )

        self.encoder = tiktoken.Encoding(
            name="p50k_pad",
            pat_str=p50k_no_special_tokens._pat_str,
            mergeable_ranks=p50k_no_special_tokens._mergeable_ranks,
            special_tokens={
                "<|endoftext|>": p50k_no_special_tokens.max_token_value + 1,
                "<|pad|>": p50k_no_special_tokens.max_token_value + 2,
            }
        )

    def encode(self, text: str) -> np.ndarray:
        text = text + "<|endoftext|>"
        text = self.encoder.encode(text, allowed_special=set(["<|endoftext|>"]))
        return text
    
    def decode(self, encoded: np.ndarray) -> str:
        return self.encoder.decode(encoded)


def as_udf(output_type) -> callable:
    def wrapper(func):
        return udf(func, output_type)
    return wrapper

encoder = SentimentAnalysisTokenizer()

PAD_TOKEN = encoder.encoder._special_tokens["<|pad|>"]

@as_udf(ArrayType(IntegerType()))
def encode_text(text):
    return encoder.encode(text)


class PipelineFunction:

    def __init__(self, func):
        self.func = func

    def __rshift__(self, other):
        def composed_func(*args, **kwargs):
            return other.func(self.func(*args, **kwargs))
        return PipelineFunction(composed_func)
    

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def __repr__(self):
        self.__call__()
        return ""

def pipeline_function(func):
    return PipelineFunction(func)





from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType, ShortType

CSV_SCHEMA = StructType([
    StructField("rating", ShortType(), nullable=False),
    StructField("title", StringType(), nullable=False),
    StructField("text", StringType(), nullable=False)
])

from functools import cache
@cache
def get_or_create_spark_session():
    return (
      SparkSession \
        .builder \
        .master("local[*]") \
        .appName("EDA") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .getOrCreate()
    )

@pipeline_function
def read_csv(path: str):
    return (
      get_or_create_spark_session()
        .read
        .schema(CSV_SCHEMA)
        .option("header", False)
        .option("mode", "FAILFAST")
        .option("inferSchema", False)
        .option("escape", '"')
        .option("multiLine", True)
        .csv(path, mode="FAILFAST")
  )

def csv(*args, **kwargs):
    @pipeline_function
    def pipe(path: str):
        return get_or_create_spark_session().read.csv(path, *args, **kwargs)
    return pipe


def repartition(*args, **kwargs):
    @pipeline_function
    def repartition(df):
        return df.repartition(*args, **kwargs)
    return repartition



def show(*args, **kwargs):
  @pipeline_function
  def show(df):
      df.show(*args, **kwargs)
      return df
  return show


def input(x):
  @pipeline_function
  def identity():
    return x
  return identity

@pipeline_function
def columns_to_numpy(rows):
    rating, text = [], []
    for row in rows:
        rating.append(row[0])
        text.append(row[1])
    return np.array(rating, dtype=np.uint8), np.array(text, dtype=np.uint16)

@pipeline_function
def print_schema(df):
    df.printSchema()
    return df

def select(*args, **kwargs):
    @pipeline_function
    def select(df):
        return df.select(*args, **kwargs)
    return select

def collect(*args, **kwargs):
    @pipeline_function
    def collect(df):
        return df.collect(*args, **kwargs)
    return collect

def take(*args, **kwargs):
    @pipeline_function
    def take(df):
        return df.take(*args, **kwargs)
    return take

def limit(*args, **kwargs):
    @pipeline_function
    def limit(df):
        return df.limit(*args, **kwargs)
    return limit

@pipeline_function
def to_numpy(df):
    return np.array(df)


def count(*args, **kwargs):
    @pipeline_function
    def count(df):
        return df.count(*args, **kwargs)
    return count


def filter(*args, **kwargs):
    @pipeline_function
    def filter(df):
        return df.filter(*args, **kwargs)
    return filter


def write_parquet(*args, **kwargs):
    @pipeline_function
    def write(df):
        df.write.parquet(*args, **kwargs)
        return df
    return write


def write(df, *args, **kwargs):
    df.write.parquet(*args, **kwargs)
    return df



@pipeline_function
def encode_dataframe_text(dataframe):
    return dataframe.select("rating", encode_text("text").alias("text"))


@pipeline_function
def preprocess_text(dataframe):
    append_fields = "case when title is null then text else concat(title, '. ', text) end as text"
    return (
      dataframe
        .select("rating", expr(append_fields).alias("text"))
        .select("rating", expr("lower(text) as text"))
    )


def read_parquet(*args, **kwargs):
    @pipeline_function
    def read_parquet(path):
        return get_or_create_spark_session().read.parquet(path, *args, **kwargs)
    return read_parquet


