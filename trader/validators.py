import numpy as np
import logging
from rltools.utils import LoggingConfig

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BookKeeper(object):
    def __init__(self, portfolio: np.ndarray):
        self.portfolio = portfolio
    
    def update(self, portfolio: np.ndarray):
        if len(portfolio) != len(self.portfolio):
            logger.error(f'Too many assets is new portfolio: # new\
                assets: {len(portfolio)}, # old assets: \
                {len(self.portfolio)}')
            raise ValueError
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
        if not action.sum() == 1.0:
            logger.error('Portfolio weights should sum to 1.')
            raise ValueError
        if action.min() < 0:
            logger.error('Shorting not allowed. All weights must \
                be larger than 0')
            raise ValueError
        return True
