import os

from data_collector import DataCollector

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), 'settings.json')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

if __name__ == '__main__':
    data_collector = DataCollector(SETTINGS_PATH)
    currencies = data_collector.settings["crypto"]
    dfs = {}
    for currency in currencies:
        dfs[currency] = data_collector.collect(currency)
        dfs[currency].to_csv(os.path.join(DATA_DIR, currency+".csv"), index=False)
