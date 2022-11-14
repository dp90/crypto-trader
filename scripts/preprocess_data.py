import os
import pandas as pd
import numpy as np
from trader.converters import MarketInterpreter
from trader.data_loader import BinanceDataLoader
from trader.indicators import collect_indicators

from configs import TradingConfig as TC, DirectoryConfig as DIR


if __name__ == "__main__":
    data_loader = BinanceDataLoader(DIR.DATA_RAW, TC)
    indicators = collect_indicators(TC)
    market_interpreter = MarketInterpreter(indicators)

    processed = np.zeros((11, 198350, 29))
    while not data_loader.is_final_time_step:
        if data_loader.index % 1000 == 0:
            print(data_loader.index)
        market_data = data_loader.next()
        statistics = market_interpreter.interpret(market_data)
        processed[:, data_loader.index - 1, :] = \
            np.hstack((market_data, statistics))

    variable_names = ['Open', 'High', 'Low', 'Close', 'Volume',
        'nTrades', 'TimeIndex']
    for indicator in indicators:
        variable_names.extend(indicator.get_indicator_names())
    
    for c, currency in enumerate(TC.CURRENCIES):
        df = pd.DataFrame(processed[c], columns=variable_names)
        df.to_csv(os.path.join(DIR.DATA_PROCESSED, f"{currency}.csv"), index=False)
