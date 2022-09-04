import numpy as np
import torch


class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.4, min_sigma=0.4, decay_period=100000):
        self.state = None
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.tensor([action + ou_state]).float()


class NormalNoiseStrategy:
    def __init__(self, action_dim, std=10, exploration_noise_ratio=0.1):
        self.action_dim = action_dim
        self.std = std
        self.exploration_noise_ratio = exploration_noise_ratio

    def reset(self):
        pass

    def get_action(self, action, t, max_exploration=False):
        if max_exploration:
            noise_scale = self.std
        else:
            noise_scale = self.exploration_noise_ratio * self.std

        noise = np.random.normal(loc=0, scale=noise_scale, size=self.action_dim)
        noisy_action = action + noise
        return torch.tensor([noisy_action]).float()


class NormalNoiseDecayStrategy:
    def __init__(self, action_dim, std=10, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=20):
        self.t = 0
        self.action_dim = action_dim
        self.std = std
        self.noise_ratio = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.decay_steps = decay_steps

    def _noise_ratio_update(self):
        noise_ratio = 1 - self.t / self.decay_steps
        noise_ratio = (self.init_noise_ratio - self.min_noise_ratio) * noise_ratio + self.min_noise_ratio
        noise_ratio = np.clip(noise_ratio, self.min_noise_ratio, self.init_noise_ratio)
        self.t += 1
        return noise_ratio

    def reset(self):
        self.t = 0

    def get_action(self, action, t, max_exploration=False):
        if max_exploration:
            noise_scale = self.std
        else:
            noise_scale = self.noise_ratio * self.std

        noise = np.random.normal(loc=0, scale=noise_scale, size=self.action_dim)
        noisy_action = action + noise

        self.noise_ratio = self._noise_ratio_update()
        return torch.tensor([noisy_action]).float()
