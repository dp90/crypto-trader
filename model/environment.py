import numpy as np
from model.data_loader import DataLoader
from typing import List


class Environment(object):
    def __init__(self, cash: str, currencies: List[str], steps_per_state: int, n_time_steps: int) -> None:
        self.cash = cash
        self.currencies = currencies
        self.steps_per_state = steps_per_state
        self.observation_space = (len(self.currencies), steps_per_state, 6)
        self.action_space = len(self.currencies)
        self.data_loader = DataLoader(self.currencies, self.steps_per_state)
        self.time = steps_per_state
        self.n_time_steps = n_time_steps
        self.portfolio_value = 1000  # Arbitrary number
        self.latest_action = np.insert(np.zeros(self.action_space), 0, 1)
        self.transaction_fee = 0.25 / 100  # == 0.25%

    def step(self, action):
        self.time += 1
        state = self.data_loader.next(self.time)  # features (OHLCVN) x assets x time_steps
        reward = self._calculate_reward(state[(0, 3), :, -2:], action)
        self.latest_action = action
        done = self._is_done()
        return state, reward, done

    def reset(self):
        self.time = self.steps_per_state
        self.portfolio_value = 1000
        self.latest_action = np.insert(np.zeros(self.action_space), 0, 1)
        return self.data_loader.next(self.time)

    def _calculate_reward(self, values, action):
        closing_portfolio_value = self._calculate_closing_portfolio_value(values, action)
        reward = closing_portfolio_value / self.portfolio_value - 1
        self.portfolio_value = closing_portfolio_value
        return reward

    def _calculate_closing_portfolio_value(self, values, action):
        prev_closing_values, opening_values, closing_values = values[3, :, -2], values[0, :, -1], values[3, :, -1]
        action = self._subtract_transaction_costs(action, prev_closing_values, opening_values)
        delta_values = np.insert(closing_values / opening_values, 0, 1)  # Add 1 to start (pos = 0) of delta_v for cash
        return np.sum(self.portfolio_value * delta_values * action)

    def _subtract_transaction_costs(self, action, prev_closing_values, opening_values):
        action += self.latest_action * (1 - opening_values / prev_closing_values)  # Account for price change at time of transactions
        # TODO: check if the line above is fair
        action -= np.abs(action - self.latest_action) * self.transaction_fee  # Subtract transaction fees
        return action

    def _is_done(self):
        out_of_data = self.time >= (self.n_time_steps - self.steps_per_state)
        agent_is_broke = self.portfolio_value < 0
        return out_of_data or agent_is_broke
