import numpy as np
import os


class DirectoryConfig(object):
    ROOT = os.path.dirname(__file__)
    VISUALS = os.path.join(ROOT, 'visuals')
    IMAGES = os.path.join(VISUALS, 'images')
    LOGS = os.path.join(ROOT, 'logs')
    RESULTS = os.path.join(LOGS, 'results')
    MODELS = os.path.join(ROOT, 'models')
    DATA = os.path.join(ROOT, 'data')


class TradingConfig:
    START_CAPITAL = 1000
    VARIABLES = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'N_TRADES']
    N_VARIABLES = len(VARIABLES)
    CURRENCY_ICS = [1, 2, 3]
    CURRENCIES = ['ADA', 'ATOM', 'BNB', 'BTC', 'ETH', 'LINK', 'LTC', 'TRX',
                  'VET', 'XLM', 'XRP']
    N_ASSETS = len(CURRENCIES)
    INITIAL_PORTFOLIO = np.zeros(len(CURRENCIES))
    INITIAL_EXCHANGE_RATE = np.zeros(len(CURRENCIES))
    OPEN_IX = 0
    HIGH_IX = 1
    LOW_IX = 2
    CLOSE_IX = 3
    VOLUME_IX = 4


class SimulationConfig:
    SLIPPAGE = 0.01
    TRANSACTION_FEE = 0.001
    OPEN_IX = 0
    CLOSE_IX = 3
