import numpy as np

from trader.validators import BookKeeper


class ActionConverter(object):
    """
    Converts actions as returned by a RL agent to HTTP request
    that the Broker can use to post to the Binance API.
    """
    def __init__(self, book_keeper: BookKeeper):
        self.book_keeper = book_keeper

    def convert(self, action: np.ndarray) -> dict:
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
        dict
            Dictionary with the HTTP request.

        Raises
        ------
        ValueError
            If requested trades are not possible with current portfolio.
        """
        self.book_keeper.is_valid(action)
        request = {}
        return request
