import numpy as np

from trader.configs import TradingConfig as TC


class BookKeeper(object):
    def __init__(self, portfolio: np.ndarray = TC.INITIAL_PF):
        self.portfolio = portfolio
    
    def update_portfolio(self, portfolio: np.ndarray):
        self.portfolio = portfolio

    def is_valid(self, action: np.ndarray) -> bool:
        """
        Checks if the suggested action by the agent is possible
        given the current portfolio. 

        Parameters
        ----------
        action : np.ndarray
            The suggested action by the agent.
        
        Returns
        -------
        bool
            Is the suggested action valid?
        """
        return False