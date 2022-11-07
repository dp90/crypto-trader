import numpy as np
import logging
from rltools.utils import LoggingConfig

from trader.data_loader import BinanceDataLoader

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BinanceSimulator:
    def __init__(self, data_loader: BinanceDataLoader, portfolio: np.ndarray):
        self.data_loader = data_loader
        self.portfolio = portfolio

    def execute(self, order):
        self._is_valid(order)
        market = self._get_market_data()
        trade = self._is_executable(order, market)
        self._update_portfolio(trade)
        return self._get_market_data(), self.portfolio

    def _is_valid(self, order) -> bool:
        # Check if requested order volumes are possible given current portfolio.
        # Throw error if not possible: converter should have modified these.
        if any((order + self.portfolio) < 0):
            logger.error(f"Sell order cannot exceed amount in portfolio: \n\
                order: {order},\nportfolio: {self.portfolio}")
            raise ValueError
        if order.sum() != 0.0:
            logger.error(f"Order should sum to 0.0, but sums to {order.sum()}")
            raise ValueError
        return True

    def _is_executable(self, order, market) -> np.ndarray:
        # Check data if order can be executed:
        # - Enough trades
        # - Enough trade volume
        # Return how much of the order can be fullfilled
        return np.zeros(1)

    def _update_portfolio(self, trade):
        # Takes latest close and adds slippage
        # Computes the new asset amounts
        # Computes the transaction costs
        # Deducts the transaction costs
        # self.portfolio = something
        pass

    def _get_market_data(self) -> np.ndarray:
        """
        Gets the next observation of the markets, including
        TIME, OPEN, HIGH, LOW, CLOSE, VOLUME, N_TRADES for each currency.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            TIME, OPEN, HIGH, LOW, CLOSE, VOLUME, N_TRADES per currency 
            in shape (n_assets, n_variables)
        """
        return self.data_loader.next()

    def reset(self):
        """
        Called from state processor. Resets:
        - Data?
        - Intial portfolio? Or should that continue from last known pf?
        """
        self.data_loader.reset()
