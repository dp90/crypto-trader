import numpy as np
from rltools.states import IStateProcessor
from rltools.utils import Scaler

from trader.converters import IOrderConverter, MarketOrderConverter, \
    LimitOrderConverter, DummyMarketInterpreter, IMarketInterpreter
from trader.data_loader import BinanceDataLoader
from trader.simulate import IBinanceSimulator, MarketOrderBinanceSimulator, \
    LimitOrderBinanceSimulator
from trader.utils import get_scale_config
from trader.validators import BookKeeper


class StateProcessor(IStateProcessor):
    def __init__(self, scaler: Scaler, binance_simulator: IBinanceSimulator,
                 action_converter: IOrderConverter, book_keeper: BookKeeper,
                 market_interpreter: IMarketInterpreter):
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
        return np.hstack((statistics.flatten(), portfolio))

    def update_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # Relevant info in state is portfolio, but that is obtained from Binance, 
        # so state function parameter is unused. 
        orders = self.converter.convert(action)
        market_data, portfolio = self.binance.execute(orders)
        self.book_keeper.update(portfolio, market_data[:, self.binance.c.CLOSE_IX])
        statistics = self.interpreter.interpret(market_data)
        return np.hstack((statistics.flatten(), portfolio))

    def reset(self) -> None:
        self.book_keeper.reset()
        self.binance.reset()


def create_hist_state_processor(trading_config, path):
    """
    Create state processor based on historic data - 
    executes market orders
    """
    data_loader = BinanceDataLoader(path, trading_config)
    binance_simulator = MarketOrderBinanceSimulator(data_loader, trading_config)
    book_keeper = BookKeeper(trading_config)
    action_converter = MarketOrderConverter(book_keeper)
    market_interpreter = DummyMarketInterpreter(trading_config)
    indicator_data = data_loader.data[:, :, trading_config.N_VARIABLES - 6:]
    scaler = Scaler(get_scale_config(trading_config, indicator_data))
    return StateProcessor(scaler, binance_simulator, action_converter,
                          book_keeper, market_interpreter)


def create_limit_order_state_processor(trading_config, path):
    """
    Create state processor based on historic data - 
    executes market orders
    """
    data_loader = BinanceDataLoader(path, trading_config)
    binance_simulator = LimitOrderBinanceSimulator(data_loader, trading_config)
    book_keeper = BookKeeper(trading_config)
    action_converter = LimitOrderConverter(book_keeper)
    market_interpreter = DummyMarketInterpreter(trading_config)
    indicator_data = data_loader.data[:, :, trading_config.N_VARIABLES - 6:]
    scaler = Scaler(get_scale_config(trading_config, indicator_data))
    return StateProcessor(scaler, binance_simulator, action_converter,
                          book_keeper, market_interpreter)
