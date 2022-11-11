import numpy as np
import logging
from rltools.utils import LoggingConfig

from trader.data_loader import BinanceDataLoader

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BinanceSimulator:
    def __init__(self, data_loader: BinanceDataLoader, portfolio: np.ndarray,
                 config):
        self.data_loader = data_loader
        self.portfolio = portfolio
        self.c = config

    def execute(self, order):
        self._is_valid(order)
        market = self.get_market_data()
        trades = self._get_trades(order, market)
        self._update_portfolio(trades)
        return market, self.portfolio

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
        raise NotImplementedError
