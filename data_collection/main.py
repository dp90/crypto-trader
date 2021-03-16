import os

from data_collection.data_collector import DataCollector

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), 'settings.json')

if __name__ == '__main__':
    data_collector = DataCollector(SETTINGS_PATH)
    tohlcv = data_collector.collect("XLM")
    print(tohlcv.head(10))



