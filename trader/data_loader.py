import pandas as pd
import numpy as np
import os
import logging
from typing import List
from rltools.utils import LoggingConfig
from trader.converters import MarketInterpreter

from trader.indicators import collect_indicators

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


class BinanceDataLoader(object):
    def __init__(self, path, config):
        self.config = config
        self.data = self._load_data(path)
        self.index = 0
        self.is_final_time_step = False

    def _load_data(self, path):
        unprocessed_assets = self._find_unprocessed_assets(path)
        if unprocessed_assets:
            self._preprocess_and_persist(unprocessed_assets, path)
        raw = self._read_data(path, self.config.CURRENCIES)
        return self._convert_data(raw)

    def _find_unprocessed_assets(self, path) -> list[str]:
        file_names = os.listdir(os.path.join(path, 'processed'))
        processed = set([file_name.split('.')[0] for file_name in file_names])
        requested = set(self.config.CURRENCIES)
        return list(requested - processed)
    
    def _preprocess_and_persist(self, path, assets):
        indicators = collect_indicators(self.config)
        variable_names = ['Open', 'High', 'Low', 'Close', 'Volume',
        'nTrades', 'TimeIndex']
        for indicator in indicators:
            variable_names.extend(indicator.get_indicator_names())
        market_interpreter = MarketInterpreter(indicators)
        path_raw = os.path.join(path, 'raw')
        path_processed = os.path.join(path, 'processed')
        raw = np.array([df.to_numpy() for df in self._read_data(path_raw, assets)])
        n_assets, n_samples, _ = raw.shape
        processed = np.zeros((n_assets, n_samples, len(variable_names)))
        for i in range(n_samples):
            if i % 1000 == 0:
                print(i)
            market_data = raw[:, i, :]
            statistics = market_interpreter.interpret(market_data)
            processed[:, i, :] = np.hstack((market_data, statistics))
        
        for a, asset in enumerate(assets):
            df = pd.DataFrame(processed[a], columns=variable_names)
            df.to_csv(os.path.join(path_processed, f"{asset}.csv"), index=False)

    def _read_data(self, path, assets):
        return [
            pd.read_csv(os.path.join(path, f"{asset}.csv"))
            for asset in assets
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

    def next(self) -> np.ndarray:
        """
        Should only be called from BinanceSimulator!!
        
        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Next OHLC-Volume-#Trades per currency in shape
            (n_assets, n_variables)
        """
        try:
            data = self.data[:, self.index, :]
        except IndexError:
            logger.error("Data loader ran out of historic data.")
            raise
        self.index += 1
        if self.index == self.data.shape[1]:
            self.is_final_time_step = True
        return data

    def reset(self):
        self.index = 0
        self.is_final_time_step = False
