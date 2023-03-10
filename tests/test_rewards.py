from trader.rewards import RewardGenerator

from configs import TestConfig as TC
from trader.validators import BookKeeper


class TestRewardGenerator:
    book_keeper = BookKeeper(TC)
    reward_generator = RewardGenerator(TC, book_keeper)

    def test_reward(self):
        pass