import requests
import pandas as pd


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
