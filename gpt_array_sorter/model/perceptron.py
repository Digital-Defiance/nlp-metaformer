import torch.nn as nn

class Perceptron(nn.Module):

    def __init__(self, model_parameters):
        super().__init__()
        self.expand_linear = nn.Linear(model_parameters.coordinates, 4 * model_parameters.coordinates, bias=False)
        self.gelu_activation = nn.GELU()
        self.project_linear = nn.Linear(4 * model_parameters.coordinates, model_parameters.coordinates, bias=False)
        self.norm = nn.LayerNorm(model_parameters.coordinates)


    def forward(self, input_tensor):
        input_tensor = self.norm(input_tensor)
        expanded_features = self.expand_linear(input_tensor)
        activated_features = self.gelu_activation(expanded_features)
        projected_features = self.project_linear(activated_features)
        return projected_features