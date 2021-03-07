from data_collector import DataCollector
from settings import SETTINGS
from python_bitvavo_api.bitvavo import Bitvavo
from datetime import datetime
import time

if __name__ == '__main__':
    # data_collector = DataCollector(SETTINGS)
    # print("Data not yet collected")

    start_string = "01/03/2020 13:24"
    start = 1000 * time.mktime(datetime.strptime(start_string, "%d/%m/%Y %H:%M").timetuple())
    end_string = "01/03/2020 14:24"
    end = 1000 * time.mktime(datetime.strptime(end_string, "%d/%m/%Y %H:%M").timetuple())

    options = {
        "start": start,
        "end": end,
    }

    bitvavo = Bitvavo()
    response = bitvavo.candles('BTC-EUR', '5m', options)
    for candle in response:
        print(datetime.fromtimestamp(candle[0] / 1000), ' open', candle[1], ' high', candle[2], ' low', candle[3],
              ' close', candle[4], ' volume', candle[5])
    remainingLimit = bitvavo.getRemainingLimit()
    print(remainingLimit)
