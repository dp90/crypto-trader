import numpy as np
from rltools.rewards import IRewardGenerator

from trader.configs import TradingConfig as TC


class RewardGenerator(IRewardGenerator):
    def __init__(self):
        self.portfolio_value = TC.START_CAPITAL

    def get_rewards(self, state: np.ndarray) -> np.ndarray:
        new_portfolio_value = state[:, TC.CURRENCY_IXS].sum()
        reward = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value
        return reward

    def reset(self):
        self.portfolio_value = TC.START_CAPITAL