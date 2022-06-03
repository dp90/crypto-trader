from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple

from configs import Mode
from ofrl.agents import IActor
from ofrl.rewards import IRewardGenerator
from ofrl.states import IStateGenerator
from ofrl.utils import IDataLoader, Scaler


class IEnvironment(ABC):
    """
    Abstract environment interface to define scenario-based environments for reinforcement learning.
    """
    def __init__(self, data_loader: IDataLoader, mode: Mode, reward_generator: IRewardGenerator, scaler: Scaler, 
                 scenario_batch_size: int = 1, n_scenarios: Optional[int] = None, seed: Optional[int] = None):
        self.data_loader = data_loader
        self.mode = mode
        self.reward_generator = reward_generator
        self.scaler = scaler
        self.scenario_batch_size = scenario_batch_size
        self.n_scenarios = n_scenarios if n_scenarios is not None else mode.n_scenarios
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.data = self._load_data()
        self.state: Optional[np.ndarray] = None

    def _load_data(self):
        """
        Loads the dataset specified with 'mode'. 
        """
        scenarios = self.data_loader.load_scenario_data(self.mode)
        return self._select_scenarios(scenarios)
    
    @abstractmethod
    def _select_scenarios(self, scenarios):
        """
        Takes a random selection of n_scenarios from the dataset to assign to the environment. If
        n_scenarios is equal to the number of scenarios in the dataset, then the dataset is shuffled 
        and returned.
        """
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the initial state, and returns it.
        """
        self.reward_generator.reset()
        self.state = self._get_initial_state()  # Is scaled
        assert self.state is not None, "'_get_initial_state()' method should have initialized state."
        return self.state
    
    @abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        """
        Creates the initial state of the environment and returns it.
        """
        # TODO: consider moving to state generator
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        Takes an action, updates the state and returns the next state, reward, done flag, and info. 
        Args:
            action: action to be taken
        Returns: 
            tuple of (state, reward, done, info)
        """
        assert self.state is not None, "Call reset before using step method."
        self.state = self._update_state(action, self.state)  # Scaled state is returned
        done = self._is_done(self.scaler.unscale(self.state))
        reward = self.reward_generator.get_rewards(self.scaler.unscale(self.state))
        info = self._get_info()
        return self.state, reward, done, info

    @abstractmethod
    def _update_state(self, action: np.ndarray, old_state: np.ndarray) -> np.ndarray:
        """
        Computes the next state of the environment as a result of taking an action. Returns
        its scaled version.
        """
        # TODO: consider moving to state generator
        raise NotImplementedError

    @abstractmethod
    def _is_done(self, state: np.ndarray) -> bool:
        """
        Checks if the environment is done. Assumes an unscaled state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_info(self) -> dict:
        """
        Returns the info dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, policy: IActor):
        """
        Evaluates the policy on the environment and returns relevant statistics. No training is done.
        Args:
            policy: policy to be evaluated
        Returns:
            tuple of (mean_reward, std_reward)
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Closes the environment.
        """
        raise NotImplementedError
