import requests
import pandas as pd


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
