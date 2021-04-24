import numpy as np
from model.data_loader import DataLoader


class Environment(object):
    def __init__(self, cash, currencies, steps_per_state, n_time_steps):
        self.cash = cash
        self.currencies = currencies
        self.steps_per_state = steps_per_state
        self.observation_space = len(self.currencies)
        self.action_space = len(self.currencies)
        self.data_loader = DataLoader(self.currencies, self.steps_per_state)
        self.time = steps_per_state
        self.portfolio_value = 1000  # Arbitrary number
        self.n_time_steps = n_time_steps

    def step(self, action):
        self.time += 1
        state = self.data_loader.next(self.time)  # assets x time_steps x features (OHLCVN)
        reward = self._calculate_reward(state[:, -1, 0], action)
        done = self._is_done()
        return state, reward, done

    def reset(self):
        self.time = self.steps_per_state
        self.portfolio_value = 1000
        return self.data_loader.next(self.time)

    def _calculate_reward(self, values_current, action):
        values_future = self.data_loader.future_open(self.time)
        delta_values = np.insert(values_future / values_current, 0, 1)  # Add 1 to the start (position = 0) of delta_v for the cash
        future_portfolio_value = self.portfolio_value * delta_values * action
        reward = future_portfolio_value / self.portfolio_value - 1
        self.portfolio_value = future_portfolio_value
        return reward

    def _is_done(self):
        # -1 for checking future opening value to calculate reward
        return self.time >= (self.n_time_steps - self.steps_per_state - 1)
