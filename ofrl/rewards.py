from abc import ABC, abstractmethod
import numpy as np


class IRewardGenerator(ABC):
    """
    Abstract reward generator interface to define reward generation for reinforcement learning.
    """
    @abstractmethod
    def get_rewards(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the reward for the given state. Assumes an unscaled state.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Resets the reward generator.
        """
        raise NotImplementedError