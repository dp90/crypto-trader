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
    N_ASSETS = 5
    VARIABLES = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'N_TRADES']
    N_VARIABLES = len(VARIABLES)
    CURRENCY_ICS = [1, 2, 3]
    CURRENCIES = ['ADA', 'ATOM', 'BNB', 'BTC', 'ETH', 'LINK', 'LTC', 'TRX',
                  'VET', 'XLM', 'XRP']
    INITIAL_PORTFOLIO = np.zeros(len(CURRENCIES))