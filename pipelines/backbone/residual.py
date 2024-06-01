

from torch import nn, Tensor

class Residual(nn.Module):
    
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    
    def forward(self, x_bcd: Tensor) -> Tensor:
        return x_bcd + self.layer(x_bcd)

