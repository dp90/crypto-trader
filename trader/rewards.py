import numpy as np
from rltools.rewards import IRewardGenerator


class RewardGenerator(IRewardGenerator):
    def __init__(self, config):
        self.config = config
        self.portfolio_value = self.config.START_CAPITAL

    def get_rewards(self, state: np.ndarray) -> np.ndarray:
        new_portfolio_value = state[:, self.config.CURRENCY_IXS].sum()
        reward = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value
        return reward

    def reset(self):
        self.portfolio_value = self.config.START_CAPITAL