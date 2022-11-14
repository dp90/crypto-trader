import numpy as np
from rltools.environments import IEnvironment
from rltools.rewards import IRewardGenerator

from trader.states import StateProcessor


class BinanceEnvironment(IEnvironment):
    def __init__(self, state_processor: StateProcessor, reward_generator: IRewardGenerator):
        super().__init__(state_processor, reward_generator)

    def _is_done(self, state: np.ndarray) -> bool:
        end_of_data = self.state_processor.binance.data_loader.is_final_time_step
        agent_is_broke = self.state_processor.book_keeper.portfolio_value <= 0
        return end_of_data or agent_is_broke

    def _get_info(self) -> dict:
        return {}
