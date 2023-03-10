import numpy as np
import os


class DirectoryConfig(object):
    ROOT = os.path.join(os.path.dirname(__file__), '..')
    VISUALS = os.path.join(ROOT, 'visuals')
    IMAGES = os.path.join(VISUALS, 'images')
    LOGS = os.path.join(ROOT, 'logs')
    RESULTS = os.path.join(LOGS, 'results')
    MODELS = os.path.join(ROOT, 'models')
    DATA = os.path.join(ROOT, 'data')
    DATA_PROCESSED = os.path.join(DATA, 'processed')
    DATA_RAW = os.path.join(DATA, 'raw')


class TradingConfig:
    START_CAPITAL = 1000
    VARIABLES = ['Open', 'High', 'Low', 'Close', 'Volume', 
        'nTrades', 'TimeIndex']
    N_VARIABLES = len(VARIABLES)
    # INDICATORS = ['RSI', 'MACD', 'MACD_EMA', 'StocOsl', 'StocOsl_SMA', 
    #     'BollingerUp', 'BollingerDown', 'EMA30', 'ADX', 'OBV_EMA', 'OBV',
    #     'Returns']
    INDICATORS = ['OBV', 'OBV_EMA', 'AwesomeOsl', 'SMA30',
        'SMA60', 'EMA30', 'EMA60', 'ADX', 'Prev_ADX', 'NegDM_EMA',
        'PosDM_EMA', 'AroonUp', 'AroonDown', 'MACD', 'MACD_EMA', 'RSI',
        'AccDist', 'BollingerUp', 'BollingerDown', 'StocOsl', 'StocOsl_SMA',
        'CCI', 'Returns']
    N_INDICATORS = len(INDICATORS)
    # CURRENCIES = ['ADA', 'ATOM', 'BNB', 'BTC', 'ETH', 'LINK', 'LTC', 'TRX',
    #               'VET', 'XLM', 'XRP']
    CURRENCIES = ['BTC', 'ETH', 'XRP']
    N_ASSETS = len(CURRENCIES)
    INITIAL_PORTFOLIO = np.array([1.0] + [0.0 for c in CURRENCIES]) * START_CAPITAL
    INITIAL_EXCHANGE_RATE = np.array([1.0] + [0.0 for c in CURRENCIES])
    OPEN_IX = 0
    HIGH_IX = 1
    LOW_IX = 2
    CLOSE_IX = 3
    VOLUME_IX = 4
    RETURNS_IX = -1
    SLIPPAGE = 0.0
    TRANSACTION_FEE = 0.0
    DATA_START_INDEX = 122_500
    DATA_FINAL_INDEX = DATA_START_INDEX + 288
