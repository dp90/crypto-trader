from trader.rewards import RewardGenerator

from configs import TestConfig as TC


class TestRewardGenerator:
    reward_generator = RewardGenerator(TC)

    def test_reward(self):
        pass