import numpy as np

from trader.environment import BinanceEnvironment
from trader.rewards import RewardGenerator
from trader.states import create_hist_state_processor
from configs import TestConfig, RESOURCES_PATH


class TestEnvironment:
    state_processor = create_hist_state_processor(TestConfig, RESOURCES_PATH)
    reward_generator = RewardGenerator(TestConfig, state_processor.book_keeper)
    environment = BinanceEnvironment(state_processor, reward_generator)

    def test_is_done(self):
        _ = self.environment.reset()
        action = np.ones(3) / 3
        _, _, done, _ = self.environment.step(action)
        assert done

    def test_get_info(self):
        assert self.environment._get_info() == {}
