from abc import ABC, abstractmethod
import os
from typing import List
import logging
import numpy as np
import pandas as pd
from rltools.utils import LoggingConfig

from trader.indicators import ITechnicalIndicator, collect_indicators
from trader.validators import BookKeeper

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class IOrderConverter(ABC):
    def __init__(self, book_keeper: BookKeeper) -> None:
        super().__init__()
        self.book_keeper = book_keeper
    
    @abstractmethod
    def convert(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MarketOrderConverter(IOrderConverter):
    """
    Converts actions as returned by a RL agent to HTTP request
    that the Broker can use to post to the Binance API.
    """
    def __init__(self, book_keeper: BookKeeper):
        super().__init__(book_keeper)

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
        action = self._format_action(action)
        self.book_keeper.are_pf_weights_valid(action)
        desired_pf_in_cash = action * self.book_keeper.get_portfolio_value()
        desired_portfolio = desired_pf_in_cash / self.book_keeper.exchange_rate
        order = desired_portfolio - self.book_keeper.portfolio
        return order

    def _format_action(self, action: np.ndarray) -> np.ndarray:
        # if action.ndim > 1:
        action = action.flatten()
        if action.sum() != 1.0:
            action = self._normalize(action)
        return action
    
    def _normalize(self, weights: np.ndarray):
        """Force sum of weights to 1.0"""
        if weights.ndim != 1:
            raise ValueError("weights array must be 1-dimensional")
        minimum = np.min(weights)
        if minimum < 0:
            weights = weights - minimum
        return weights / np.linalg.norm(weights, ord=1)


class LimitOrderConverter(IOrderConverter):
    def __init__(self, book_keeper: BookKeeper):
        super().__init__(book_keeper)
        self.c = self.book_keeper.config

    def convert(self, action):
        """
        Converts action as returned by a RL agent to limit orders.
        Rows are assets and contain 4 values:
        - Column 0: amount to buy
        - Column 1: price to buy at
        - Column 2: amount to sell
        - Column 3: price to sell at

        Parameters
        ----------
        action : np.ndarray
            Action from RL agent.
        
        Returns
        -------
        np.ndarray
            Amount and price to buy at, and amount and price to sell at
            per asset in shape (n_assets, 4)
        """
        order = np.maximum(action, 0.).reshape((-1, 4))
        order[:, 1] = np.where(order[:, 1] > self.book_keeper.exchange_rate[1:],
                               1.5 * self.book_keeper.exchange_rate[1:], 
                               order[:, 1])
        buy_amount = (order[:, 0] @ order[:, 1]) * (1 + self.c.TRANSACTION_FEE)
        # Can't buy more than there is cash available
        if buy_amount > self.book_keeper.portfolio[0]:
            order[:, 0] = order[:, 0] * (self.book_keeper.portfolio[0] / buy_amount)
        # Can't sell more than there is:
        order[:, 2] = np.minimum(order[:, 2], self.book_keeper.portfolio[1:])
        return order


def asset_dfs_to_numpy(dfs):
    return np.array([df.to_numpy() for df in dfs])


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.market_interpreter = MarketInterpreter(collect_indicators(self.config))

    def preprocess(self, path):
        processed_path = os.path.join(path, 'processed')
        unprocessed_assets = self._find_unprocessed_assets(processed_path)
        if unprocessed_assets:
            self._preprocess_and_persist(unprocessed_assets, path)
    
    def _find_unprocessed_assets(self, processed_path) -> list[str]:
        file_names = os.listdir(processed_path)
        processed = set([file_name.split('.')[0] for file_name in file_names])
        requested = set(self.config.CURRENCIES)
        return list(requested - processed)

    def _preprocess_and_persist(self, assets, path):
        orig_n_assets = self.config.N_ASSETS
        self.config.N_ASSETS = len(assets)
        self.config.N_ASSETS = orig_n_assets
        variable_names = self.config.VARIABLES + \
            self.market_interpreter.get_indicator_names()
        path_raw = os.path.join(path, 'raw')
        path_processed = os.path.join(path, 'processed')
        raw = [
            pd.read_csv(os.path.join(path_raw, f"{asset}.csv"))
            for asset in assets
        ]
        processed = self._convert_data(raw)
        
        for a, asset in enumerate(assets):
            df = pd.DataFrame(processed[a], columns=variable_names)
            df.to_csv(os.path.join(path_processed, f"{asset}.csv"), index=False)

    def _convert_data(self, data: List[pd.DataFrame]) -> np.ndarray:
        raw = self._time_to_time_index(data)
        raw = asset_dfs_to_numpy(raw)
        processed = self._add_indicators(raw)
        return processed

    def _time_to_time_index(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
        for df in data:
            df['TIME'] = pd.to_datetime(df['TIME'])
            df['TIME_INDEX'] = df['TIME'].dt.hour * 12 + df['TIME'].dt.minute // 5
            df.drop(columns='TIME', inplace=True)
        return data

    def _add_indicators(self, raw: np.ndarray) -> np.ndarray:
        variable_names = self.config.VARIABLES + \
            self.market_interpreter.get_indicator_names()
        n_assets, n_samples, _ = raw.shape
        processed = np.zeros((n_assets, n_samples, len(variable_names)))
        for i in range(n_samples):
            if i % 1000 == 0:
                logger.info(f"Preprocessing technical indicators for sample "
                            f"{i + 1} / {n_samples}")
            market_data = raw[:, i, :]
            statistics = self.market_interpreter.interpret(market_data)
            processed[:, i, :] = np.hstack((market_data, statistics))
        return processed


class IMarketInterpreter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def interpret(self, market_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_indicator_names(self) -> list[str]:
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
            
    def get_indicator_names(self) -> list[str]:
        names = []
        for indicator in self.indicators:
            names.extend(indicator.get_indicator_names())
        return names


class DummyMarketInterpreter(IMarketInterpreter):
    def __init__(self, trading_config):
        self.config = trading_config

    def interpret(self, market_data: np.ndarray) -> np.ndarray:
        # N_VARIABLES - 3 will include Volume, nTrades and TimeIndex
        return market_data[:, self.config.N_VARIABLES - 6:]

    def reset(self):
        pass
    
    def get_indicator_names(self) -> list[str]:
        return self.config.INDICATORS
