from os.path import join, dirname, realpath
import numpy as np

RESOURCES_PATH = join(dirname(realpath(__file__)), 'resources')


class TestConfig:
    START_CAPITAL = 1000
    N_ASSETS = 5
    VARIABLES = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'N_TRADES']
    N_VARIABLES = len(VARIABLES)
    CURRENCY_ICS = [1, 2, 3]
    CURRENCIES = ['ADA', 'ATOM']
    INITIAL_PORTFOLIO = np.array([10, 25, 3])