import numpy as np


class OpalConfig(object):
    """
    Holds variable definitions for the OPAL environment.
    """
    N_ASSETS = 7
    SCENARIO_DURATION = 40
    INITIAL_WEIGHTS = np.array([0.8 * 0.8, 0.8 * 0.2, 0.1 * 0.8, 0.1 * 0.2, 0, 0.05, 0.05])
    START_CAPITAL = 100_000
    GOAL_WEALTH = 400_000  # In real terms
    TRANSACTION_COSTS = 0.005