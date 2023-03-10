import pandas as pd
import numpy as np
import os
import logging
from rltools.utils import LoggingConfig

from trader.converters import Preprocessor, asset_dfs_to_numpy
from trader.indicators import collect_indicators

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BinanceDataLoader(object):
    def __init__(self, path, config):
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.data = self._load_data(path)
        self.reset()

    def _load_data(self, path):
        self.preprocessor.preprocess(path)
        asset_dfs = self._read_processed_data(path, self.config.CURRENCIES)
        return asset_dfs_to_numpy(asset_dfs)
    
    def _read_processed_data(self, path, assets):
        processed_path = os.path.join(path, 'processed')
        dfs = self._read_data(processed_path, assets)
        if self.config.INDICATORS:
            dfs = [self._filter_indicators(df) for df in dfs]
        return dfs

    def _filter_indicators(self, df: pd.DataFrame):
        return df[self.config.VARIABLES + self.config.INDICATORS]

    def _read_data(self, path, assets):
        return [
            pd.read_csv(os.path.join(path, f"{asset}.csv"))
            for asset in assets
        ]

    def next(self) -> np.ndarray:
        """
        Should only be called from BinanceSimulator!!
        
        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Next OHLC-Volume-#Trades and indicators per 
            currency in shape 
            (n_assets, n_variables + n_indicators)
        """
        try:
            data = self.data[:, self.index, :]
        except IndexError:
            logger.error("Data loader ran out of historic data.")
            raise
        self.index += 1
        if self.index == self.config.DATA_FINAL_INDEX: #self.data.shape[1]:
            self.is_final_time_step = True
        return data

    def reset(self):
        self.index = self.config.DATA_START_INDEX
        self.is_final_time_step = False
