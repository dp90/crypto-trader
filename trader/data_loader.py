import pandas as pd
import numpy as np
import os
import logging
from typing import List
from rltools.utils import LoggingConfig

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BinanceDataLoader(object):
    def __init__(self, path, config):
        self.config = config
        self.data = self._load_data(path)
        self.index = 0

    def _load_data(self, path):
        raw = self._read_data(path)
        return self._convert_data(raw)

    def _read_data(self, path):
        return [
            pd.read_csv(os.path.join(path, currency + ".csv"))
            for currency in self.config.CURRENCIES
        ]
    
    def _convert_data(self, data: List[pd.DataFrame]) -> np.ndarray:
        """
        Converts a list of Pandas DataFrames to a single 3D numpy array

        Parameters
        ----------
        data : List[pd.DataFrames]
            Raw OHLC-Volume-#Trades-TimeOfDay data
        
        Returns
        -------
        np.ndarray
            OHLC-Volume-#Trades-TimeIndex per currency in the 
            shape (n_assets, time_steps, n_variables)
        """
        for df in data:
            df['TIME'] = pd.to_datetime(df['TIME'])
            df['TIME_INDEX'] = df['TIME'].dt.hour * 12 + df['TIME'].dt.minute // 5
            df.drop(columns='TIME', inplace=True)
        return np.array([df.to_numpy() for df in data])

    def next(self):
        try:
            data = self.data[:, self.index, :]
        except IndexError:
            logger.error("Data loader ran out of historic data.")
            raise
        self.index += 1
        return data

    def reset(self):
        self.index = 0
