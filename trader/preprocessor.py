class Preprocessor(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize(state):
        state[:, :, :4] = state[:, :, :4] / state[:, -1, 3]  # Normalize OHLC to C of latest time step for each asset
        state[:, :, 4] = state[:, :, 4] / state[:, -1, 4]  # Normalize V to V of latest time step for each asset
        state[:, :, 5] = state[:, :, 5] / state[:, -1, 5]  # Normalize N to N of latest time step for each asset
        return state
