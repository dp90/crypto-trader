import numpy as np
from rltools.states import IStateProcessor
from rltools.utils import Scaler

from trader.converters import ActionConverter, MarketInterpreter
from trader.simulate import BinanceSimulator
from trader.validators import BookKeeper


class StateProcessor(IStateProcessor):
    def __init__(self, scaler: Scaler, binance_simulator: BinanceSimulator,
                 action_converter: ActionConverter, book_keeper: BookKeeper,
                 market_interpreter: MarketInterpreter):
        super().__init__(scaler)
        self.binance = binance_simulator  # Statefull
        self.converter = action_converter  # Stateless
        self.interpreter = market_interpreter  # Stateless
        self.book_keeper = book_keeper  # Statefull
    
    def get_initial_state(self) -> np.ndarray:
        market_data = self.binance.get_market_data()
        statistics = self.interpreter.interpret(market_data)
        portfolio = self.binance.portfolio
        self.book_keeper.update(portfolio, market_data[:, self.binance.c.CLOSE_IX])
        return np.concatenate((statistics, portfolio))

    def update_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # Relevant info in state is portfolio, but that is obtained from Binance, 
        # so state function parameter is unused. 
        orders = self.converter.convert(action)
        market_data, portfolio = self.binance.execute(orders)
        self.book_keeper.update(portfolio, market_data[:, self.binance.c.CLOSE_IX])
        statistics = self.interpreter.interpret(market_data)
        return np.concatenate((statistics, portfolio))

    def reset(self) -> None:
        self.binance.reset()
        # self.book_keeper.reset()
