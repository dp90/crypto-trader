from abc import ABC, abstractmethod
from typing import List
import numpy as np

from trader.indicators import ITechnicalIndicator
from trader.validators import BookKeeper


class ActionConverter(object):
    """
    Converts actions as returned by a RL agent to HTTP request
    that the Broker can use to post to the Binance API.
    """
    def __init__(self, book_keeper: BookKeeper):
        self.book_keeper = book_keeper

    def convert(self, action: np.ndarray) -> np.ndarray:
        """
        Converts actions as returned by a RL agent to buy and sell
        orders in the form of an HTTP request that can be sent to 
        the Binance API.

        Parameters
        ----------
        action : np.ndarray
            Action from RL agent.
        
        Returns
        -------
        np.ndarray
            Amounts to buy and sell in terms of cash. Negative
            values are sells, positive buys.

        Raises
        ------
        ValueError
            If requested trades are not possible with current portfolio.
        """
        self.book_keeper.is_valid(action)
        desired_pf_in_cash = action * self.book_keeper.get_portfolio_value()
        desired_portfolio = desired_pf_in_cash / self.book_keeper.exchange_rate
        order = desired_portfolio - self.book_keeper.portfolio
        return order


class IMarketInterpreter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def interpret(self, market_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError


class MarketInterpreter(IMarketInterpreter):
    def __init__(self, indicators: list[ITechnicalIndicator]):
        self.indicators = indicators

    def interpret(self, market_data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        market_data : np.ndarray
            OHLC-Volume-#Trades in shape (n_assets, n_variables)

        Returns
        -------
        np.ndarray
            Technical indicators
        """
        interpretation = []
        for indicator in self.indicators:
            indicator.interpret(market_data)
            interpretation.append(indicator.get_indicators())
        return np.hstack(interpretation)
    
    def reset(self):
        for indicator in self.indicators:
            indicator.reset()


class DummyMarketInterpreter(IMarketInterpreter):
    def __init__(self, trading_config):
        self.config = trading_config

    def interpret(self, market_data: np.ndarray) -> np.ndarray:
        return market_data[self.config.N_VARIABLES:]

    def reset(self):
        pass
    