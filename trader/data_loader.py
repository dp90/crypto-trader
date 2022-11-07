import pandas as pd
import numpy as np
import os
from typing import List

from trader.configs import TradingConfig as TC

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


class BinanceDataLoader(object):
    def __init__(self, path, config = TC):
        self.data = self._load_data(path)
        self.config = config
        self.index = 0

    def _load_data(self, path):
        raw = self._read_data(path)
        return self._convert_data(raw)

    def _read_data(self, path):
        return [
            pd.read_csv(os.path.join(path, currency + ".csv"))
            for currency in TC.CURRENCIES
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
            df = df.drop(columns='TIME')
        return np.array([df.to_numpy() for df in data])

    def next(self):
        data = self.data[:, self.index, :]
        self.index += 1
        return data

    def reset(self):
        self.index = 0


class DataLoader(object):
    def __init__(self, currencies, steps_per_state):
        self.currencies = currencies
        self.steps_per_state = steps_per_state
        self.data = np.array([
            pd.read_csv(os.path.join(DATA_DIR, currency + ".csv")).drop(columns=["TIME"]).to_numpy()
            for currency in self.currencies
        ])
        # TODO: Summarize older data (e.g. statistics over first 4h of last 24h, stats over 3rd 2h, 4th 2h, etc.) ->
        #  refine towards recent data

    def next(self, time):
        return np.transpose(self.data[:, time - self.steps_per_state: time, :], (2, 0, 1))  # F x M x T
