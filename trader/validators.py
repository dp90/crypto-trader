import numpy as np
import logging
from rltools.utils import LoggingConfig
import math

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BookKeeper(object):
    def __init__(self, trading_config):
        self.config = trading_config
        self.reset()
    
    def get_portfolio_value(self):
        return self.portfolio @ self.exchange_rate
    
    def update(self, portfolio: np.ndarray, exchange_rate: np.ndarray):
        if len(portfolio) != len(self.portfolio):
            logger.error(f'Too many assets is new portfolio: # new\
                assets: {len(portfolio)}, # old assets: \
                {len(self.portfolio)}')
            raise ValueError
        if len(exchange_rate) + 1 != len(self.exchange_rate):
            logger.error(f'Too many assets is new exchange_rate: # new\
                exchange_rate: {len(exchange_rate)}, # old \
                exchange_rate: {len(self.exchange_rate)}')
            raise ValueError
        self.portfolio = portfolio.copy()
        self.exchange_rate[1:] = exchange_rate  # Idx 0 (USDT) stays 1

    def are_pf_weights_valid(self, action: np.ndarray) -> bool:
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
        if not math.isclose(action.sum(), 1.0):
            logger.error('Portfolio weights should sum to 1.')
            raise ValueError
        if action.min() < 0:
            logger.error('Shorting not allowed. All weights must \
                be larger than 0')
            raise ValueError
        return True

    def reset(self):
        self.portfolio = self.config.INITIAL_PORTFOLIO.copy()
        self.exchange_rate = self.config.INITIAL_EXCHANGE_RATE.copy()
    