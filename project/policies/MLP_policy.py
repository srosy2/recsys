import abc
from abc import ABC
from typing import Any
from torch import nn
from torch import optim

import numpy as np
import torch

from project.infrastructure import pytorch_util as ptu
from project.policies.base_policy import BasePolicy
from torch.distributions import Normal


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 hidden_dim,
                 learning_rate=1e-5,
                 decay=1e-6,
                 std=False,
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
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.hidden_dim,
        )
        self.mean_net.to('cpu')
        if std:
            self.logstd = nn.Parameter(torch.zeros(self.ac_dim))
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.decay
        )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, user, memory, state_repr,
                   action_emb,
                   items=None,
                   return_scores=False
                   ):
        if items is None:
            items = torch.tensor([i for i in range(state_repr.item_embeddings.weight.shape[0])])
        scores = torch.bmm(state_repr.item_embeddings(items).unsqueeze(0),
                           action_emb.T.unsqueeze(0)).squeeze(0)
        if return_scores:
            return scores, torch.gather(items, 0, scores.argmax(0))
        else:
            return torch.gather(items, 0, scores.argmax(0))

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        return self.mean_net(observation)

class MLPA2CPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 net, device="cpu",
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.net = net
        self.device = device

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        states_v = observation.to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd
        return torch.tensor(actions)


class MLPSACPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 hidden_dim,
                 learning_rate=1e-5,
                 decay=1e-6,
                 entropy_lr=0.001,
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

        self.mean_net = ptu.build_sac_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.hidden_dim,
        )
        self.mean_net.to('cpu')
        self.optimizer = optim.Adam(
            self.mean_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.decay
        )

        self.target_entropy = -np.prod(ac_dim)
        self.logalpha = torch.zeros(1, requires_grad=True, device='cpu')
        self.alpha_optimizer = optim.Adam([self.logalpha], lr=entropy_lr)

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, user, memory, state_repr,
                   action_emb,
                   items=None,
                   return_scores=False
                   ):
        if items is None:
            items = torch.tensor([i for i in range(state_repr.item_embeddings.weight.shape[0])])
        scores = torch.bmm(state_repr.item_embeddings(items).unsqueeze(0),
                           action_emb.T.unsqueeze(0)).squeeze(0)
        if return_scores:
            return scores, torch.gather(items, 0, scores.argmax(0))
        else:
            return torch.gather(items, 0, scores.argmax(0))

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        return self.mean_net(observation)

    def full_pass(self, state):
        mean, log_std = self.forward(state)

        pi_s = Normal(mean, log_std.exp())
        action = pi_s.rsample()

        log_prob = pi_s.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def select_action(self, state):
        mean, log_std = self.forward(state)

        action = Normal(mean, log_std.exp()).sample()
        return action
