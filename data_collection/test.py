import os
import pandas as pd

from data_collection.binance import Binance
from data_collection.main import DATA_DIR
import requests

from data_collection.utils import TimeConverter

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(DATA_DIR, "ADA.csv"))
    incorrect_data = pd.read_csv(os.path.join(DATA_DIR, "incorrect_ADA.csv"))

    time_converter = TimeConverter("%d/%m/%Y %H:%M", "Binance")
    start_interval = time_converter.to_timestamp("15/05/2019", "04:01")
    end_interval = time_converter.to_timestamp("15/05/2019", "06:01")

    binance = Binance()

    results = binance.candles("ADAUSDT", '5m', {'start': start_interval, 'end': end_interval})
    results.TIME = results.TIME.apply(lambda timestamp: time_converter.from_timestamp(timestamp))