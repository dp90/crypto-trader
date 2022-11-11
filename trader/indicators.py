import numpy as np
from abc import ABC, abstractmethod


class ITechnicalIndicator(ABC):
    def __init__(self, config):
        self.c = config

    @abstractmethod
    def interpret(self, market_data):
        raise NotImplementedError
    
    @abstractmethod
    def get_indicators(self) -> list:
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

    def get_indicators(self) -> list:
        return [self.obv, self.obv_ema]

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

    def get_indicators(self) -> list:
        return [self.ema5 - self.ema34]

    def reset(self):
        self.ema34 = np.zeros(self.c.N_ASSETS)
        self.ema5 = np.zeros(self.c.N_ASSETS)


class SimpleMovingAverage(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()

    def interpret(self, market_data):
        new = market_data[:, self.c.CLOSE_IX]
        self.sma30 = self.sma30 + (new - self.sma30) / 30
        self.sma60 = self.sma60 + (new - self.sma60) / 60

    def get_indicators(self) -> list:
        return [self.sma30, self.sma60]

    def reset(self):
        self.sma30 = np.zeros(self.c.N_ASSETS)
        self.sma60 = np.zeros(self.c.N_ASSETS)


class ExponentialMovingAverage(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        n_time_steps, smoothing = 20, 2
        self.k = smoothing / (n_time_steps + 1)
    
    def interpret(self, market_data):
        new = market_data[:, self.c.CLOSE_IX]
        self.ema30 = new * self.k + self.ema30 * (1 - self.k)
        self.ema60 = new * self.k + self.ema60 * (1 - self.k)

    def get_indicators(self) -> list:
        return [self.ema30, self.ema60]
    
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
            market_data[:, self.c.HIGH_IX] - self.market[:, self.c.CLOSE_IX],
            market_data[:, self.c.CLOSE_IX] - self.market[:, self.c.LOW_IX],
        ])
        self.sma_TR = self.sma_TR + (TR - self.sma_TR) / 14
    
    def _update_ema_directional_movement(self, market_data):
        dm_pos = market_data[:, self.c.HIGH_IX] - self.market[:, self.c.HIGH_IX]
        dm_neg = market_data[:, self.c.LOW_IX] - self.market[:, self.c.LOW_IX]
        self.ema_DM_pos = dm_pos * self.k + self.ema_DM_pos * (1 - self.k)
        self.ema_DM_neg = dm_neg * self.k + self.ema_DM_neg * (1 - self.k)
    
    def _update_market_data(self, market_data):
        self.market = market_data

    def _update_average_directional_index(self):
        di_pos = 100 * self.ema_DM_pos / self.sma_TR
        di_neg = 100 * self.ema_DM_neg / self.sma_TR
        adx = 100 * np.abs(di_pos - di_neg) / np.abs(di_pos + di_neg)
        self.prev_adx = self.adx.copy()
        self.adx = adx * self.k + self.adx * (1 - self.k)

    def get_indicators(self) -> list:
        return [self.adx, self.prev_adx, self.ema_DM_neg, self.ema_DM_pos]

    def reset(self):
        self.market = np.zeros((self.c.N_ASSETS, self.c.N_VARIABLES))
        self.sma_TR = np.zeros(self.c.N_ASSETS)
        self.ema_DM_pos = np.zeros(self.c.N_ASSETS)
        self.ema_DM_neg = np.zeros(self.c.N_ASSETS)
        self.prev_adx = np.zeros(self.c.N_ASSETS)
        self.adx = np.zeros(self.c.N_ASSETS)


class Aroon(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def interpret(self, market_data):
        self.high = np.concatenate((self.high[:, 1:], market_data[:, self.c.HIGH_IX]), axis=1)
        self.low = np.concatenate((self.low[:, 1:], market_data[:, self.c.LOW_IX]), axis=1)
        self.aroon_up = (np.nanargmax(self.high) + 1) * 4  # = * 100 / 25
        self.aroon_down = (np.nanargmin(self.low) + 1) * 4  # = * 100 / 25

    def get_indicators(self) -> list:
        return [self.aroon_up, self.aroon_down]

    def reset(self):
        self.high = np.full((self.c.N_ASSETS, 25), np.nan)
        self.low = np.full((self.c.N_ASSETS, 25), np.nan)
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

    def get_indicators(self) -> list:
        return [self.macd, self.ema_macd]
    
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
        self.ema_up_change = (up + 13 * self.ema_up_change) * (1 / 14)
        self.ema_down_change = (down + 13 * self.ema_down_change) * (1 / 14)
        RS = self.ema_up_change / self.ema_down_change
        self.rsi = 100 - 100 / (1 + RS)

    def get_indicators(self) -> list:
        return [self.rsi]

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

    def get_indicators(self) -> list:
        return [self.ad]

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
        old_sma_tp = self.sma_tp.copy()
        self.sma_tp = self.sma_tp + (TP - self.sma_tp) / 20
        self.dn2 = self.dn2 + (TP - self.sma_tp) * (TP - old_sma_tp)

    def get_indicators(self) -> list:
        std_tp = np.sqrt(self.dn2 / 20)
        bol_up = self.sma_tp + 2 * std_tp
        bol_down = self.sma_tp - 2 * std_tp
        return [bol_up, bol_down]

    def reset(self):
        self.sma_tp = np.zeros(self.c.N_ASSETS)
        self.dn2 = np.zeros(self.c.N_ASSETS)


class StochasticOscillator(ITechnicalIndicator):
    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def interpret(self, market_data):
        closing_price = market_data[:, self.c.CLOSE_IX]
        l14 = np.min(self.low14, axis=1)
        h14 = np.max(self.high14, axis=1)
        self.indicator = 100 * (closing_price - l14) / (h14 - l14)
        self.sma = self.sma + (self.indicator - self.sma) / 3
        ix = self.counter % 14
        self.low14[:, ix] = market_data[:, self.c.LOW_IX]
        self.high14[:, ix] = market_data[:, self.c.HIGH_IX]
        self.counter += 1

    def get_indicators(self) -> list:
        return [self.indicator, self.sma]

    def reset(self):
        self.counter = 0
        self.indicator = np.zeros(self.c.N_ASSETS)
        self.sma = np.zeros(self.c.N_ASSETS)
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

        self.sma_tp = self.sma_tp + (TP - self.sma_tp) / 20
        diff = TP - self.sma_tp
        self.sma_diff = self.sma_diff + np.abs(diff - self.sma_diff) / 20
        self.cci = diff / (0.015 * self.sma_diff)

    def get_indicators(self) -> list:
        return [self.cci]
    
    def reset(self):
        self.sma_tp = np.zeros(self.c.N_ASSETS)
        self.sma_diff = np.zeros(self.c.N_ASSETS)
        self.cci = np.zeros(self.c.N_ASSETS)
