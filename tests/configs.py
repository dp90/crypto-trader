from os.path import join, dirname, realpath
import numpy as np

RESOURCES_PATH = join(dirname(realpath(__file__)), 'resources')


class TestConfig:
    START_CAPITAL = 1000
    VARIABLES = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'N_TRADES', 'TIME_INDEX']
    N_VARIABLES = len(VARIABLES)
    INDICATORS = []
    N_INDICATORS = len(INDICATORS)
    CURRENCIES = ['ADA', 'ATOM']
    N_ASSETS = len(CURRENCIES)
    INITIAL_PORTFOLIO = np.array([10., 25., 3.])
    INITIAL_EXCHANGE_RATE = np.array([1.0, 0.06804, 3.925])  # 1st asset is cash
    OPEN_IX = 0
    HIGH_IX = 1
    LOW_IX = 2
    CLOSE_IX = 3
    VOLUME_IX = 4
    SLIPPAGE = 0.01
    TRANSACTION_FEE = 0.001
    DATA_START_INDEX = 0
    DATA_FINAL_INDEX = 2
