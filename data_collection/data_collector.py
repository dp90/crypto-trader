import time

from settings_reader import SettingsReader
from python_bitvavo_api.bitvavo import Bitvavo
import pandas as pd
from utils import TimeConverter
import requests


class DataCollector(object):
    def __init__(self, settings_path):
        self.settings = SettingsReader.read(settings_path)
        self.time_converter = TimeConverter(self.settings)
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


class Poloniex(object):
    def __init__(self):
        self.base = "https://poloniex.com/public"
        self.min2sec_map = {'5m': 300, '15m': 900, "30m": 1800}

    def candles(self, currency_pair, time_step, options):
        query_params = {
            "command": "returnChartData",
            "currencyPair": currency_pair,
            "start": options['start'],
            "end": options['end'],
            "period": self.min2sec_map.get(time_step, 1440)
        }
        response = requests.get(url=self.base, params=query_params)
        df = pd.DataFrame(response.json())
        df = df.drop(columns=['quoteVolume', 'weightedAverage'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df = df.rename(columns={'date': 'TIME', 'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'CLOSE',
                                'volume': 'VOLUME'})
        return df


class Binance(object):
    def __init__(self):
        self.base = "https://api1.binance.com"
        self.response_elements = ["TIME", "OPEN", "HIGH", "LOW", "CLOSE", "base volume", "close time", "VOLUME",
                                  "NUMBER_OF_TRADES", "taker buy base volume", "taker buy quote asset volume", "ignore"]
        self.ignore_elements = ["close time", "base volume", "taker buy base volume", "taker buy quote asset volume",
                                "ignore"]
        self.exceeded_rate_limit = False

    def candles(self, currency_pair, time_step, options):
        url = "/api/v3/klines"
        query_params = {
            "symbol": currency_pair,
            "startTime": options['start'],
            "endTime": options['end'],
            "interval": time_step
        }
        if not self.exceeded_rate_limit:
            response = requests.get(url=self.base + url, params=query_params)
            if response.status_code == 429:
                self.exceeded_rate_limit = True
            df = pd.DataFrame(response.json(), columns=self.response_elements)
            df = df.drop(columns=self.ignore_elements)
            return df
        else:
            raise RuntimeError("Binance rate limit is exceeded.")


exchange_map = {'Binance': Binance(), 'Poloniex': Poloniex(), 'Bitvavo': Bitvavo()}
