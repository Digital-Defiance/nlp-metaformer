
from pydantic import BaseModel
from torch import nn



class TestLayer(BaseModel, nn.Module):
  a: int
  b: int
