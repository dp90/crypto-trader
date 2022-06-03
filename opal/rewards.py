import numpy as np
from ofrl.rewards import IRewardGenerator

from opal.opal_config import OpalConfig as OPAL
from opal.utils import DataLoader, get_portfolio_returns


class RewardGeneratorD(IRewardGenerator):
    """
    Generates rewards with Group D's reward function, containing
    - 1 if the final wealth is greater than the target wealth
    - The fraction of the target wealth attained, if target wealth not achieved
    - The downside risk
    """
    def __init__(self, reward_params) -> None:
        historic_returns = np.dot(DataLoader().load_historic_data()[:, 0:7], OPAL.INITIAL_WEIGHTS)
        self.ret_ema = np.mean(historic_returns)
        self.ddsq = np.mean(np.power(np.minimum(historic_returns, 0), 2))
        self.reward_params = reward_params
    
    def get_rewards(self, state: np.ndarray) -> np.ndarray:
        goal_wealth = OPAL.GOAL_WEALTH
        asset_returns, weights_current, realized_wealth, time_left = np.hsplit(state, [8, 15, 16])
        done = (time_left == 0)
        net_returns = get_portfolio_returns(asset_returns, weights_current)

        #Binary reward
        binary_reward = self._get_binary_rewards(done, realized_wealth, goal_wealth)

        # Penalty for violating asset bandwidth constraints
        soft_constraint_penalty = self._get_soft_constraint_penalty(weights_current)

        #Loss given failure
        lgf_reward = self._get_loss_given_failure_rewards(realized_wealth, goal_wealth, done)

        #Update the ewma of the returns and DD for the DDR
        net_returns = net_returns - 1
        new_ret_ema = self.ret_ema + np.multiply(self.reward_params['adaptation_rate'], (net_returns - self.ret_ema))
        new_DDsq = self.ddsq + np.multiply(self.reward_params['adaptation_rate'], (np.power(np.minimum(net_returns, 0), 2)) - self.ddsq)

        #Compute DDR
        downside_risk_reward = np.divide((new_ret_ema), (np.sqrt(new_DDsq))).reshape(-1)
        self.ret_ema = new_ret_ema
        self.ddsq = new_DDsq

        #Full reward function specification
        reward = (binary_reward + \
            self.reward_params["rwf_dist_1"] * lgf_reward) - \
            self.reward_params["rwf_dist_3"] * soft_constraint_penalty + \
            self.reward_params["rwf_dist_4"] * downside_risk_reward

        return reward

    def _get_binary_rewards(self, done: np.ndarray, realized_wealth: np.ndarray, goal_wealth: float) -> np.ndarray:
        return np.multiply(done, realized_wealth >= goal_wealth).squeeze()

    def _get_loss_given_failure_rewards(self, realized_wealth: np.ndarray, goal_wealth: float, done: np.ndarray) -> np.ndarray:
        LGF = np.divide(realized_wealth - goal_wealth, goal_wealth).squeeze() + 1
        LGF -= np.multiply(LGF - 1, LGF > 1)
        LGF += np.multiply(-LGF, LGF < 0)
        LGF = np.multiply(LGF, np.multiply(done, realized_wealth < goal_wealth).squeeze())
        return LGF

    def _get_soft_constraint_penalty(self, weights_current: np.ndarray) -> np.ndarray:
        # Initialize distance to 0
        distance = np.zeros(weights_current.shape[0])

        # Equity between 0.2 and 0.8
        distance += np.multiply(0.2 - np.sum(weights_current[:, 0:2], axis=1),
                                np.sum(weights_current[:, 0:2], axis=1) < 0.2)
        distance += np.multiply(np.sum(weights_current[:, 0:2], axis=1) - 0.8,
                                np.sum(weights_current[:, 0:2], axis=1) > 0.8)

        # Fixed income between 0.2 and 0.8
        distance += np.multiply(0.2 - np.sum(weights_current[:, 2:4], axis=1),
                                np.sum(weights_current[:, 2:4], axis=1) < 0.2)
        distance += np.multiply(np.sum(weights_current[:, 2:4], axis=1) - 0.8,
                                np.sum(weights_current[:, 2:4], axis=1) > 0.8)

        # Separate Equity and Fixed Income assets greater than 0
        distance += np.multiply(- weights_current[:, 0], weights_current[:, 0] < 0)
        distance += np.multiply(- weights_current[:, 1], weights_current[:, 1] < 0)
        distance += np.multiply(- weights_current[:, 2], weights_current[:, 2] < 0)
        distance += np.multiply(- weights_current[:, 3], weights_current[:, 3] < 0)

        # Cash between 0 and 0.3
        distance += np.multiply(- weights_current[:, 4], weights_current[:, 4] < 0)
        distance += np.multiply(weights_current[:, 4] - 0.3, weights_current[:, 4] > 0.3)

        # Real estate between 0 and 0.2
        distance += np.multiply(- weights_current[:, 5], weights_current[:, 5] < 0)
        distance += np.multiply(weights_current[:, 5] - 0.2, weights_current[:, 5] > 0.2)

        # Hedge funds between 0 and 0.1
        distance += np.multiply(- weights_current[:, 6], weights_current[:, 6] < 0)
        distance += np.multiply(weights_current[:, 6] - 0.1, weights_current[:, 6] > 0.1)

        # Change distances as wanted
        distance += np.multiply(self.reward_params["rwf_dist_2"], distance > 0)
        distance -= np.multiply(distance - 10, distance > 10)
        distance = distance / 10
        return distance

    def reset(self):
        historic_returns = np.dot(DataLoader().load_historic_data()[:, 0:7], OPAL.INITIAL_WEIGHTS)
        self.ret_ema = np.mean(historic_returns)
        self.ddsq = np.mean(np.power(np.minimum(historic_returns, 0), 2))