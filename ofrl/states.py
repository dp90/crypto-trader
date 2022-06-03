from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from configs import IMode, Mode

from ofrl.utils import IDataLoader


class IStateGenerator(ABC):
    """
    Interface to define state generators for scenario-based reinforcement learning.
    """
    def __init__(self, mode: Mode, data_loader: IDataLoader, scenario_batch_size: int = 1, 
                 n_scenarios: Optional[int] = None, seed: Optional[int] = None):
        self.mode = mode
        self.data_loader = data_loader
        self.scenario_batch_size = scenario_batch_size
        self.n_scenarios = n_scenarios if n_scenarios is not None else mode.n_scenarios
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.data = self._load_data()

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

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """
        Generates the initial state of the environment and returns it.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Generates the next state given the current state and action.
        """
        raise NotImplementedError
