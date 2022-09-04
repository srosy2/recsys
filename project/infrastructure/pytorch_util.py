from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network
        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    class NN(nn.Module):
        def __init__(self):
            super(NN, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(input_size, size)] + [nn.Linear(size, size) for _ in range(2,
                                                                                                               n_layers)
                                                                          ] + [nn.Linear(size, output_size)])

        def initialize(self):
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)

        def forward(self, x):
            for num in range(len(self.linears) - 1):
                x = activation(self.linears[num](x))

            x = output_activation(self.linears[-1](x))

            return x

    return NN()


def build_sac_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network
        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    class NN(nn.Module):
        def __init__(self):
            super(NN, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(input_size, size)] + [nn.Linear(size, size) for _ in range(2,
                                                                                                               n_layers)
                                                                          ])
            self.output_layer_mean = nn.Linear(size, output_size)
            self.output_layer_log_std = nn.Linear(size, output_size)

        def initialize(self):
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)

        def forward(self, x):
            for num in range(len(self.linears)):
                x = activation(self.linears[num](x))
            x_mean = output_activation(self.output_layer_mean(x))
            x_log_std = output_activation(self.output_layer_log_std(x))

            return x_mean, x_log_std

    return NN()


device = 'cpu'


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to(device).detach().numpy()
