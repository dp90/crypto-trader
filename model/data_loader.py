import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


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
        return self.data[:, time - self.steps_per_state: time, :]  # M x T x F

    def future_open(self, time):
        return self.data[:, time: time + 1, 0]