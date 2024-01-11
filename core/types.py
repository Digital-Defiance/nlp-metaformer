
from torch import Tensor
from typing import Annotated
from pydantic.functional_validators import AfterValidator


TensorInt = Tensor

TensorFloat = Tensor

def greater_than_zero(value: int) -> int:
    assert value > 0, "Value must be greater than zero"
    return value

PositiveInt = Annotated[int, AfterValidator(greater_than_zero)]
