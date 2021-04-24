import time
import pandas as pd
import requests
import os

from utils.settings_reader import SettingsReader
from utils import TimeConverter

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), 'settings.json')

if __name__ == "__main__":
    base = "https://api1.binance.com"
    response = requests.get(base + "/api/v3/ticker/price").json()
    symbols = [result['symbol'] for result in response]
    usdtPairs = [symbol for symbol in symbols if symbol[-4:] == "USDT"]
    invalidPairs = ["BCCUSDT", "VENUSDT", "BCHABCUSDT", 'BCHSVUSDT', 'USDSUSDT']
    usdtPairs = [pair for pair in usdtPairs if pair not in invalidPairs]

    settings = SettingsReader.read(SETTINGS_PATH)
    converter = TimeConverter(settings)
    start = converter.to_timestamp(settings['startDate'], settings['startTime'])

    pair_volumes = []
    for i, pair in enumerate(usdtPairs):
        print("Getting", i + 1, "out of", len(usdtPairs), "pairs:", pair)
        url = "/api/v3/klines"
        query_params = {
            "symbol": pair,
            "startTime": start,
            "interval": "1M"
        }
        http_response = requests.get(url=base + url, params=query_params)
        try:
            for response in http_response.json()[-4:-1]:
                pair_volumes.append([pair, converter.from_timestamp(response[0]), converter.from_timestamp(response[6]),
                                     response[7], response[8]])
        except:
            invalidPairs.append(pair)
        time.sleep(1)
        if http_response.status_code == 429:
            break

    volume_df = pd.DataFrame(pair_volumes, columns=["PAIR", "START_TIME", "END_TIME", "VOLUME", "N_TRADES"])
    volume_df = volume_df.sort_values(by="VOLUME")
    volume_df.to_csv("volume_per_pair", index=False)
    print(volume_df.sort_values(by=["VOLUME", "N_TRADES"]).head(15))

    data = pd.read_csv("../data/volume_per_pair")
    data = data.sort_values(by="VOLUME")
    data = data.drop(columns=["START_TIME", "END_TIME"])
    grouped = data.groupby(["PAIR"]).mean()

    grouped = grouped.sort_values(by=["VOLUME"])