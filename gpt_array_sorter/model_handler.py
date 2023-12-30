from model import NanoGPT
import mlflow
import mlflow.pytorch
from pydantic_settings import BaseSettings
from system_parameters import DEVICE
from dotenv import load_dotenv

load_dotenv()

class ModelHandler(BaseSettings):
    """
    Represents the parameters of a model.

    Attributes:
        coordinates (int): The dimension of a vector embedding.
        tokens (int): The number of tokens in the vocabulary.
        words (int): The maximum number of words in a sentence (context window).
        number_of_blocks (int): The number of blocks in the model.
    """

    coordinates: int = 3*3
    tokens: int = 3
    words: int = 11
    number_of_blocks: int = 3


    @classmethod
    def export_parameters(cls):
        return cls().model_dump_json()

    @classmethod
    def create_from_parameters(cls, json_data: str):
        json = json.loads(json_data)
        params = cls(**json)
        nanoGPT = NanoGPT(params)
        nanoGPT.params = params
        mlflow.log_param("number_of_blocks", params.number_of_blocks)
        mlflow.log_param("coordinates", params.coordinates)
        mlflow.log_param("tokens", params.tokens)
        mlflow.log_param("words", params.words)
        mlflow.log_param("parameters", nanoGPT.count_parameters())
        mlflow.log_param("device", DEVICE)
        return nanoGPT.to(DEVICE)
