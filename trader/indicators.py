import numpy as np
from abc import ABC, abstractmethod

from trader.utils import shift_window


class ITechnicalIndicator(ABC):
    def __init__(self, config):
        self.c = config

    @abstractmethod
    def interpret(self, market_data):
        raise NotImplementedError
    
    @abstractmethod
    def get_indicators(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError


class OnBalanceVolume(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()
        N = 20
        self.k = 2 / (N + 1)

    def interpret(self, market_data):
        self._on_balance_volume(market_data)
        self._obv_ema()

    def get_indicators(self) -> np.ndarray:
        return np.column_stack((self.obv, self.obv_ema))

    def _on_balance_volume(self, market_data):
        growth_sign = np.sign(market_data[:, self.c.CLOSE_IX] - \
            self.last_close)
        self.obv += growth_sign * market_data[:, self.c.VOLUME_IX]
        self.last_close = market_data[:, self.c.CLOSE_IX].copy()

    def _obv_ema(self):
        """
        Compute exponential moving average of on-balance volume
        """
        self.obv_ema = self.obv * self.k + self.obv_ema * (1 - self.k)

    def reset(self):
        self.last_close = np.zeros(self.c.N_ASSETS)
        self.obv = np.zeros(self.c.N_ASSETS)
        self.obv_ema = np.zeros(self.c.N_ASSETS)


class AwesomeOscillator(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def interpret(self, market_data):
        mid = (market_data[:, self.c.OPEN_IX] + \
               market_data[:, self.c.OPEN_IX]) / 2
        self.ema34 = self.ema34 + (mid - self.ema34) / 34
        self.ema5 = self.ema5 + (mid - self.ema5) / 5

    def get_indicators(self) -> np.ndarray:
        return (self.ema5 - self.ema34)[:, None]

    def reset(self):
        self.ema34 = np.zeros(self.c.N_ASSETS)
        self.ema5 = np.zeros(self.c.N_ASSETS)


class SimpleMovingAverage(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def interpret(self, market_data):
        self.hist = shift_window(self.hist, market_data[:, self.c.CLOSE_IX])

    def get_indicators(self) -> np.ndarray:
        sma30 = self.hist[:, -30:].mean(axis=1)
        sma60 = self.hist.mean(axis=1)
        return np.column_stack((sma30, sma60))

    def reset(self):
        self.hist = np.zeros((self.c.N_ASSETS, 60))


class ExponentialMovingAverage(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def interpret(self, market_data):
        new = market_data[:, self.c.CLOSE_IX]
        self.ema30 = (new + 29 * self.ema30) / 30
        self.ema60 = (new + 59 * self.ema60) / 60

    def get_indicators(self) -> np.ndarray:
        return np.column_stack((self.ema30, self.ema60))
    
    def reset(self):
        self.ema30 = np.zeros(self.c.N_ASSETS)
        self.ema60 = np.zeros(self.c.N_ASSETS)


class AverageDirectionalIndex(ITechnicalIndicator):
    """
    TR: true range
    DM: directional movement
    DI: directional index
    """
    def __init__(self, config):
        super().__init__(config)
        self.k = 1 / 15
        self.reset()
    
    def interpret(self, market_data):
        self._update_sma_true_range(market_data)
        self._update_ema_directional_movement(market_data)
        self._update_market_data(market_data)
        self._update_average_directional_index()

    def _update_sma_true_range(self, market_data):
        TR = np.maximum.reduce([
            market_data[:, self.c.HIGH_IX] - market_data[:, self.c.LOW_IX],
            np.abs(market_data[:, self.c.HIGH_IX] - self.market[:, self.c.CLOSE_IX]),
            np.abs(market_data[:, self.c.CLOSE_IX] - self.market[:, self.c.LOW_IX]),
        ])
        self.TR_hist = shift_window(self.TR_hist, TR)
    
    def _update_ema_directional_movement(self, market_data):
        dm_pos = np.maximum(market_data[:, self.c.HIGH_IX] \
                            - self.market[:, self.c.HIGH_IX], 0)
        dm_neg = np.maximum(self.market[:, self.c.LOW_IX] \
                            - market_data[:, self.c.LOW_IX], 0)
        dm_pos = np.where(dm_pos < dm_neg, 0, dm_pos)
        dm_neg = np.where(dm_neg < dm_pos, 0, dm_neg)
        
        self.ema_DM_pos = dm_pos * self.k + self.ema_DM_pos * (1 - self.k)
        self.ema_DM_neg = dm_neg * self.k + self.ema_DM_neg * (1 - self.k)
    
    def _update_market_data(self, market_data):
        self.market = market_data

    def _update_average_directional_index(self):
        sma_TR = self.TR_hist.mean(axis=1)
        di_pos = 100 * self.ema_DM_pos / sma_TR
        di_neg = 100 * self.ema_DM_neg / sma_TR
        dx = 100 * np.abs(di_pos - di_neg) / (di_pos + di_neg)
        self.prev_adx = self.adx.copy()
        self.adx = dx * self.k + self.adx * (1 - self.k)

    def get_indicators(self) -> np.ndarray:
        return np.column_stack((self.adx, self.prev_adx, self.ema_DM_neg, self.ema_DM_pos))

    def reset(self):
        self.market = np.zeros((self.c.N_ASSETS, self.c.N_VARIABLES))
        self.TR_hist = np.zeros((self.c.N_ASSETS, 14))
        self.ema_DM_pos = np.zeros(self.c.N_ASSETS)
        self.ema_DM_neg = np.zeros(self.c.N_ASSETS)
        self.prev_adx = np.zeros(self.c.N_ASSETS)
        self.adx = np.zeros(self.c.N_ASSETS)


class Aroon(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def interpret(self, market_data):
        self.high = shift_window(self.high, market_data[:, self.c.HIGH_IX])
        self.low = shift_window(self.low, market_data[:, self.c.LOW_IX])
        self.aroon_up = (np.nanargmax(self.high, axis=1)) * 4  # = * 100 / 25
        self.aroon_down = (np.nanargmin(self.low, axis=1)) * 4  # = * 100 / 25

    def get_indicators(self) -> np.ndarray:
        return np.column_stack((self.aroon_up, self.aroon_down))

    def reset(self):
        self.high = np.full((self.c.N_ASSETS, 26), np.nan)
        self.low = np.full((self.c.N_ASSETS, 26), np.nan)
        self.aroon_up = np.zeros(self.c.N_ASSETS)
        self.aroon_down = np.zeros(self.c.N_ASSETS)


class MovingAverageConvergenceDivergence(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def interpret(self, market_data):
        close = market_data[:, self.c.CLOSE_IX]
        self.ema_12 = (close + 11 * self.ema_12) * (1 / 12)
        self.ema_26 = (close + 25 * self.ema_26) * (1 / 26)
        self.macd = self.ema_12 - self.ema_26
        self.ema_macd = (self.macd + 8 * self.ema_macd) * (1 / 9)

    def get_indicators(self) -> np.ndarray:
        return np.column_stack((self.macd, self.ema_macd))
    
    def reset(self):
        self.macd = np.zeros(self.c.N_ASSETS)
        self.ema_12 = np.zeros(self.c.N_ASSETS)
        self.ema_26 = np.zeros(self.c.N_ASSETS)
        self.ema_macd = np.zeros(self.c.N_ASSETS)


class RelativeStrengthIndex(ITechnicalIndicator):
    """
    RSI: relative strength index
    RS: relative strength
    """
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def interpret(self, market_data):
        new_close = market_data[:, self.c.CLOSE_IX]
        up = np.maximum(new_close - self.latest_close, 0)
        down = np.maximum(self.latest_close - new_close, 0)
        self.latest_close = new_close
        self.ema_up_change = (up + 13 * self.ema_up_change) * (1 / 14)
        self.ema_down_change = (down + 13 * self.ema_down_change) * (1 / 14)
        RS = self.ema_up_change / self.ema_down_change
        self.rsi = 100 - 100 / (1 + RS)

    def get_indicators(self) -> np.ndarray:
        return self.rsi[:, None]

    def reset(self):
        self.latest_close = np.zeros(self.c.N_ASSETS)
        self.ema_up_change = np.zeros(self.c.N_ASSETS)
        self.ema_down_change = np.zeros(self.c.N_ASSETS)
        self.rsi = np.zeros(self.c.N_ASSETS)


class AccumulationDistribution(ITechnicalIndicator):
    """
    CMFV: current money flow volume
    """
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def interpret(self, market_data):
        MFM = ((market_data[:, self.c.CLOSE_IX] - market_data[:, self.c.LOW_IX])\
            - (market_data[:, self.c.HIGH_IX] - market_data[:, self.c.CLOSE_IX]))\
            / (market_data[:, self.c.HIGH_IX] - market_data[:, self.c.LOW_IX])
        CMFV = MFM * market_data[:, self.c.VOLUME_IX]
        self.ad = self.ad + CMFV

    def get_indicators(self) -> np.ndarray:
        return self.ad[:, None]

    def reset(self):
        self.ad = np.zeros(self.c.N_ASSETS)


class BollingerBands(ITechnicalIndicator):
    """
    TP: typical price
    """
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def interpret(self, market_data):
        TP = np.mean(
            market_data[:, [self.c.HIGH_IX, self.c.LOW_IX, self.c.CLOSE_IX]],
            axis=1)
        self.tp_hist = shift_window(self.tp_hist, TP)

    def get_indicators(self) -> np.ndarray:
        std_tp = np.std(self.tp_hist, axis=1)
        sma_tp = np.mean(self.tp_hist, axis=1)
        bol_up = sma_tp + 2 * std_tp
        bol_down = sma_tp - 2 * std_tp
        return np.column_stack((bol_up, bol_down))  # Consider adding the difference between the 2

    def reset(self):
        self.tp_hist = np.zeros((self.c.N_ASSETS, 20))


class StochasticOscillator(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def interpret(self, market_data):
        self._update_hist(market_data)
        closing_price = market_data[:, self.c.CLOSE_IX]
        l14 = np.min(self.low14, axis=1)
        h14 = np.max(self.high14, axis=1)
        self.indicator = 100 * (closing_price - l14) / (h14 - l14)
        self.indicator_hist = shift_window(self.indicator_hist, self.indicator)

    def _update_hist(self, market_data):
        ix = self.counter % 14
        self.counter += 1
        self.low14[:, ix] = market_data[:, self.c.LOW_IX]
        self.high14[:, ix] = market_data[:, self.c.HIGH_IX]

    def get_indicators(self) -> np.ndarray:
        sma = self.indicator_hist.mean(axis=1)
        return np.column_stack((self.indicator, sma))

    def reset(self):
        self.counter = 0
        self.indicator_hist = np.zeros((self.c.N_ASSETS, 3))
        self.indicator = np.zeros(self.c.N_ASSETS)
        self.low14 = np.zeros((self.c.N_ASSETS, 14))
        self.high14 = np.zeros((self.c.N_ASSETS, 14))


class CommodityChannelIndex(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def interpret(self, market_data):
        TP = np.mean(
            market_data[:, [self.c.HIGH_IX, self.c.LOW_IX, self.c.CLOSE_IX]],
            axis=1)
        self.tp_hist = shift_window(self.tp_hist, TP)
        diff = TP - self.tp_hist.mean(axis=1)
        self.diff_hist = shift_window(self.diff_hist, diff)
        self.cci = diff * (1 / (0.015 * np.abs(self.diff_hist).mean(axis=1)))

    def get_indicators(self) -> np.ndarray:
        return self.cci[:, None]
    
    def reset(self):
        self.tp_hist = np.zeros((self.c.N_ASSETS, 20))
        self.diff_hist = np.zeros((self.c.N_ASSETS, 20))
        self.cci = np.zeros(self.c.N_ASSETS)
