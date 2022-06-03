from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np

from ofrl.utils import mlp


class IActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int], activation: nn.Module, 
                 output_activation: nn.Module) -> None:
        super().__init__()
        self.network = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

    def _distribution(self, obs):
        raise NotImplementedError

    def forward(self, obs):
        raise NotImplementedError
    
    def act(self, obs):
        raise NotImplementedError


class PpoActor(IActor):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...], activation: nn.Module, 
                 output_activation: nn.Module) -> None:
        super().__init__(obs_dim, act_dim, hidden_sizes, activation, output_activation)

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def act(self, obs):
        with torch.no_grad():
            pi = self._distribution(obs)
            a = pi.sample()
            logp_a = self._log_prob_from_distribution(pi, a)
        return a.numpy(), logp_a.numpy()


class PpoCategoricalActor(PpoActor):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int], activation: nn.Module, 
                 output_activation: nn.Module):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation, output_activation)

    def _distribution(self, obs):
        logits = self.network(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class PpoGaussianActor(PpoActor):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...], activation: nn.Module, 
                 output_activation: nn.Module, action_log_std: float):
        super().__init__(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        log_std = action_log_std * np.ones(act_dim, dtype=np.float64)
        self.log_std = nn.parameter.Parameter(torch.as_tensor(log_std))

    def _distribution(self, obs):
        mu = self.network(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class IValue(nn.Module):
    """
    Value function interface that inherits from torch.nn.Module.
    """
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...], activation: nn.Module, 
                 output_activation: nn.Module):
        super().__init__()
        self.network = mlp([obs_dim] + list(hidden_sizes) + [1], activation, output_activation)

    @abstractmethod
    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Args:
            state: state to map to a value
        Returns: 
            value - expected cumulative discounted reward
        """
        raise NotImplementedError


class MLPCritic(IValue):
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...], activation: nn.Module, 
                 output_activation: nn.Module):
        super().__init__(obs_dim, hidden_sizes, activation, output_activation)
        
    def forward(self, obs):
        return torch.squeeze(self.network(obs), -1)  # Squeeze to ensure v has right shape.