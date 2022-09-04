import abc
from typing import Any
from torch import nn
from torch import optim

import numpy as np
import torch

from project.infrastructure import pytorch_util as ptu
from project.values.base_value import BaseValue


class MLPValue(BaseValue, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 hidden_dim,
                 learning_rate=1e-4,
                 decay=1e-6,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.decay = decay

        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=1,
            n_layers=self.n_layers, size=self.hidden_dim,
        )
        self.mean_net.to('cpu')
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.decay
        )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor, action: torch.FloatTensor) -> Any:
        x = torch.cat([observation, action], 1)
        return self.mean_net(x)


class MLPPPOValue(BaseValue, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ob_dim,
                 n_layers,
                 hidden_dim,
                 learning_rate=1e-4,
                 decay=1e-6,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.decay = decay

        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=1,
            n_layers=self.n_layers, size=self.hidden_dim,
        )
        self.mean_net.to('cpu')
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.decay
        )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        x = torch.cat([observation], 1)
        return self.mean_net(x)


#####################################################
#####################################################
class MLPTD3Value(BaseValue, nn.Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 hidden_dim,
                 learning_rate=1e-4,
                 decay=1e-6,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.decay = decay

        self.mean_net_1 = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=1,
            n_layers=self.n_layers, size=self.hidden_dim,
        )
        self.mean_net_2 = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=1,
            n_layers=self.n_layers, size=self.hidden_dim,
        )
        self.mean_net_1.to('cpu')
        self.mean_net_2.to('cpu')
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.decay
        )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor, action: torch.FloatTensor) -> Any:
        x = torch.cat([observation, action], 1)
        return self.mean_net_1(x), self.mean_net_2(x)

    def Qa(self, observation, action):
        x = torch.cat([observation, action], 1)
        return self.mean_net_1(x)
