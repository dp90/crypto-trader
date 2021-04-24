import time

from data_collection.poloniex import Poloniex
from data_collection.binance import Binance
from utils.settings_reader import SettingsReader
from python_bitvavo_api.bitvavo import Bitvavo
import pandas as pd
from data_collection.utils import TimeConverter


class DataCollector(object):
    def __init__(self, settings_path):
        self.settings = SettingsReader.read(settings_path)
        self.time_converter = TimeConverter(self.settings["timeFormat"], self.settings["exchange"])
        self.exchange = exchange_map[self.settings['exchange']]

    def collect(self, currency):
        final_time = self.time_converter.to_timestamp(self.settings['endDate'], self.settings['endTime'])
        start_interval = self.time_converter.to_timestamp(self.settings['startDate'], self.settings['startTime'])
        dataframes = []

        while start_interval < final_time:
            print(currency, "- interval starting at:", self.time_converter.from_timestamp(start_interval))
            end_interval = self.time_converter.add_hours(start_interval, self.settings['request_interval_in_hours'])
            if end_interval > final_time:
                end_interval = final_time

            options = {
                "start": start_interval,
                "end": end_interval
            }
            currency_pair = self.get_currency_pair(currency)
            response = self.exchange.candles(currency_pair, self.settings['timeStep'], options)

            if type(response) == pd.DataFrame:
                tohlcv_columns = list(response.columns)
            else:
                tohlcv_columns = ["TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]

            tohlcv = pd.DataFrame(response, columns=tohlcv_columns)
            tohlcv.TIME = tohlcv.TIME.apply(lambda timestamp: self.time_converter.from_timestamp(timestamp))
            dataframes.append(tohlcv)
            start_interval = end_interval
            time.sleep(1)

        return pd.concat(dataframes)

    def get_currency_pair(self, currency):
        exchange = self.settings['exchange']
        if exchange == "Bitvavo":
            return currency + "-" + self.settings['cash']
        elif exchange == "Poloniex":
            return self.settings['cash'] + "_" + currency
        elif exchange == "Binance":
            return currency + self.settings['cash']


exchange_map = {'Binance': Binance(), 'Poloniex': Poloniex(), 'Bitvavo': Bitvavo()}
