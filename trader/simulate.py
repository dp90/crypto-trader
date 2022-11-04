import numpy as np

from configs import TradingConfig as TC


class BinanceSimulator(object):
    def __init__(self, data_loader, portfolio):
        self.data_loader = data_loader
        self.portfolio = portfolio

    def execute(self, trade):
        self.is_valid(trade)
        market = self.load_market()
        self.is_executable(trade, market)
        self.update_portfolio(trade)
        return self.get_market_data(), self.get_portfolio()

    def is_valid(self, trade) -> bool:
        # Check if requested trade volumes are possible given current portfolio.
        # Throw error if not possible: converter should have modified these.
        return False

    def load_market(self):
        # Loads the next trading period of data
        pass

    def is_executable(self, trade, market) -> np.ndarray:
        # Check data if trade can be executed:
        # - Enough trades
        # - Enough trade volume
        pass

    def update_portfolio(self, trade):
        # Takes a price in between high and low
        # Computes the new asset amounts
        # Computes the transaction costs
        # Deducts the transaction costs
        pass

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
        return np.zeros((TC.N_ASSETS, TC.N_VARIABLES))

    def get_portfolio(self) -> np.ndarray:
        """
        Gets the next portfolio balance: amount in cash for each currency.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            amount in cash per currency
        """
        return np.zeros(TC.N_ASSETS)