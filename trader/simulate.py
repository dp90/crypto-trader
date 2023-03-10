import numpy as np
import logging
from abc import ABC, abstractmethod
from rltools.utils import LoggingConfig

from trader.data_loader import BinanceDataLoader

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class IBinanceSimulator(ABC):
    def __init__(self, data_loader: BinanceDataLoader, config):
        self.data_loader = data_loader
        self.c = config
        self.portfolio = self.c.INITIAL_PORTFOLIO.copy()

    def execute(self, order):
        self._is_valid(order)
        market = self.get_market_data()
        trades = self._get_trades(order, market)
        self._update_portfolio(trades)
        return market, self.portfolio

    @abstractmethod
    def _is_valid(self, order) -> bool:
        """
        Checks if requested order volumes are possible given 
        current portfolio. Throws error if not possible: 
        converter should have modified these.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_trades(self, order, market) -> np.ndarray:
        """
        Uses the requested order and the market to derive
        which trades are executed.
        """
        raise NotImplementedError

    def _update_portfolio(self, trades):
        self.portfolio += trades

    def get_market_data(self) -> np.ndarray:
        """
        Gets the next observation of the markets, including
        TIME, OPEN, HIGH, LOW, CLOSE, VOLUME, N_TRADES for each currency.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            TIME, OPEN, HIGH, LOW, CLOSE, VOLUME, N_TRADES, TimeIndex 
            and indicators per currency in shape 
            (n_assets, n_variables + n_indicators)
        """
        return self.data_loader.next()

    def reset(self):
        """
        Called from state processor. Resets:
        - Data?
        - Intial portfolio? Or should that continue from last known pf?
        """
        self.data_loader.reset()
        self.portfolio = self.c.INITIAL_PORTFOLIO.copy()


class MarketOrderBinanceSimulator(IBinanceSimulator):
    def __init__(self, data_loader: BinanceDataLoader, config):
        super().__init__(data_loader, config)
    
    def _is_valid(self, order) -> bool:
        # Check if requested order volumes are possible given current portfolio.
        # Throw error if not possible: converter should have modified these.
        if any((order + self.portfolio) < 0):
            logger.error(f"Sell order cannot exceed amount in portfolio: \n\
                order: {order},\nportfolio: {self.portfolio}")
            raise ValueError
        return True

    def _get_trades(self, order, market) -> np.ndarray:
        """
        Uses the opening market price as a reference: slippage and transaction
        fees are added to that price. Negative costs are an addition, so that
        negative costs are reduced with slippage and fees, while positive ones
        are increased.
        """
        price = market[:, self.c.OPEN_IX]
        cash_cost = order[1:] * price
        cash_cost = np.where(cash_cost > 0,
                             cash_cost * (1 + self.c.SLIPPAGE),
                             cash_cost * (1 - self.c.SLIPPAGE))
        cash_cost = np.where(cash_cost > 0,
                             cash_cost * (1 + self.c.TRANSACTION_FEE),
                             cash_cost * (1 - self.c.TRANSACTION_FEE))
        trades = np.zeros_like(order)
        trades[1:] = order[1:]
        trades[0] -= cash_cost.sum()
        return trades


class LimitOrderBinanceSimulator(IBinanceSimulator):
    """ Executes limit orders """

    def __init__(self, data_loader: BinanceDataLoader, config):
        super().__init__(data_loader, config)
    
    def _is_valid(self, order) -> bool:
        # Selling more of a ccy than available is not allowed
        # Buying accommodated as long as cash is available
        return True

    def _get_trades(self, order, market) -> np.ndarray:
        buy = np.where(order[:, 1] < market[:, self.c.LOW_IX], 0., order[:, 0])
        sell = np.where(order[:, 3] > market[:, self.c.HIGH_IX], 0., order[:, 2])
        cash_cost = (buy @ order[:, 1]) * (1 + self.c.TRANSACTION_FEE) - \
                    (sell @ order[:, 3]) * (1 - self.c.TRANSACTION_FEE)
        trades = np.zeros_like(self.portfolio)
        trades[1:] = buy - sell
        trades[0] = -cash_cost
        return trades
