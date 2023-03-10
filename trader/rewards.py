import numpy as np
from rltools.rewards import IRewardGenerator

from trader.validators import BookKeeper


class RewardGenerator(IRewardGenerator):
    def __init__(self, config, book_keeper: BookKeeper):
        self.config = config
        self.book_keeper = book_keeper
        self.reset()

    def get_rewards(self, state: np.ndarray) -> np.ndarray:
        # weights = state[-(self.config.N_ASSETS + 1):]
        # punishment = (weights[weights>1] - 1).sum() - weights[weights<0].sum()
        pf_value = self.book_keeper.get_portfolio_value()
        returns = 1 + (pf_value - self.prev_pf_value) / self.prev_pf_value
        self.prev_pf_value = pf_value
        downside_risk_reward = self.downside_risk_reward(returns)
        # np.log(returns) + downside_risk_reward  - 0.0005 * punishment
        rewards = np.array([np.log(returns)])
        return rewards

    def downside_risk_reward(self, returns: float) -> float:
        # Update the ewma of the returns and DD for the DDR
        net_returns = returns - 1
        new_ret_ema = self.ret_ema + 0.04 * (net_returns - self.ret_ema)
        new_DDsq = self.ddsq + 0.04 * (np.minimum(net_returns, 0)**2 - self.ddsq)

        downside_risk_reward = new_ret_ema / np.sqrt(new_DDsq)
        self.ret_ema = new_ret_ema
        self.ddsq = new_DDsq
        return downside_risk_reward

    def reset(self):
        self.prev_pf_value = self.config.START_CAPITAL
        self.ret_ema = 0.001
        self.ddsq = 0.001
