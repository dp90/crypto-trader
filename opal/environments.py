import numpy as np
from typing import Optional

import torch

from configs import Mode
from ofrl.rewards import IRewardGenerator
from opal.utils import get_transaction_costs, normalize_weights, update_old_weights, update_portfolio_value
from opal.opal_config import OpalConfig as OPAL
from ofrl.agents import IActor
from ofrl.environments import IEnvironment
from ofrl.utils import IDataLoader, Scaler


class OpalEnv(IEnvironment):
    def __init__(self, data_loader: IDataLoader, mode: Mode, reward_generator: IRewardGenerator, scaler: Scaler,
                 scenario_batch_size: int = 1, n_scenarios: Optional[int] = None, seed: Optional[int] = None):
        super().__init__(data_loader, mode, reward_generator, scaler, scenario_batch_size, n_scenarios, seed)
        self.historic_data = data_loader.load_historic_data()
        self.batch_index = 0
        self.max_batch_index = self.data.shape[1] // self.scenario_batch_size \
            if self.data.shape[1] % self.scenario_batch_size != 0 else self.data.shape[1] // self.scenario_batch_size - 1
        
    def _get_initial_state(self) -> np.ndarray:
        elapsed_time = 0

        idx_start = self.batch_index * self.scenario_batch_size
        idx_end = min(self.mode.n_scenarios, (self.batch_index + 1) * self.scenario_batch_size)
        inflation = self.data[elapsed_time, idx_start: idx_end, -1][:, np.newaxis]
        last_year_return = np.tile(self.historic_data[-1, 0:7], (self.scenario_batch_size, 1))
        last_year_return = np.concatenate((last_year_return, inflation), axis=1)

        weights = OPAL.INITIAL_WEIGHTS
        weights_matrix = np.tile(weights.T, (self.scenario_batch_size, 1))
        time_left = (OPAL.SCENARIO_DURATION - elapsed_time) * np.ones((self.scenario_batch_size, 1))

        portfolio_start = OPAL.START_CAPITAL * np.ones((self.scenario_batch_size, 1))
        state = np.concatenate((last_year_return, weights_matrix, portfolio_start, time_left), axis=1)

        # TODO: is it necessary to put self.state in another ndarray? Isn't it already an ndarray?
        return self.scaler.scale(np.array(state))

    def _update_state(self, action: np.ndarray, state: np.ndarray):
        state = self.scaler.unscale(state)
        prev_returns, _, prev_weights_old, prev_portfolio_value, prev_time_left = np.hsplit(state, [7, 8, 15, 16])
        elapsed_time = OPAL.SCENARIO_DURATION - prev_time_left[0, 0] + 1
        
        new_returns = self.data[int(elapsed_time), self.batch_index * self.scenario_batch_size:(self.batch_index + 1) * self.scenario_batch_size, :]
        
        prev_weights = update_old_weights(prev_returns, prev_weights_old)  # Weights change during a time step due to market returns
        new_weights = normalize_weights(action)
        transaction_costs = get_transaction_costs(prev_weights, new_weights, prev_portfolio_value, OPAL.TRANSACTION_COSTS)
        prev_portfolio_value -= transaction_costs

        new_portfolio_value = update_portfolio_value(new_returns, new_weights, prev_portfolio_value)
        time_left = (OPAL.SCENARIO_DURATION - elapsed_time) * np.ones((self.scenario_batch_size, 1))
        return self.scaler.scale(np.concatenate((new_returns, new_weights, new_portfolio_value, time_left), axis=1))

    def _is_done(self, state: np.ndarray) -> np.ndarray:
        time_left = state[:, -1]
        done = (time_left == 0).any()
        if done:
            scenario_set_done = self.batch_index == self.max_batch_index
            if scenario_set_done:
                self._shuffle_scenarios()
                self.batch_index = 0
            else:
                self.batch_index += 1
        return done

    def _get_info(self):
        return {}

    def evaluate(self, actor: IActor):
        raise NotImplementedError

    def _shuffle_scenarios(self):
        self.data = self.rng.permutation(self.data, axis=1)

    def _select_scenarios(self, scenarios):
        selection = self.rng.choice(self.mode.n_scenarios, self.n_scenarios, replace=False)
        return scenarios[:, selection, :]

    def close(self):
        raise NotImplementedError