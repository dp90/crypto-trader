import numpy as np

from tests.configs import TestConfig
from trader.indicators import (Aroon, 
                               AverageDirectionalIndex, 
                               AwesomeOscillator, BollingerBands, CommodityChannelIndex, 
                               ExponentialMovingAverage, MovingAverageConvergenceDivergence, 
                               OnBalanceVolume, RelativeStrengthIndex, 
                               SimpleMovingAverage, StochasticOscillator)


class TestOnBalanceVolume:
    indicator = OnBalanceVolume(TestConfig)

    def test_interpret(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator.interpret(market)
        try:
            np.testing.assert_allclose(self.indicator.last_close, np.array([0.068, 3.914]))
            np.testing.assert_allclose(self.indicator.obv, 
                                    np.array([11313.6248 ,  1254.06378]))
            np.testing.assert_allclose(self.indicator.obv_ema, 
                                    np.array([1077.48807619,  119.43464571]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator.interpret(market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], 
                                    np.array([11313.6248 ,  1254.06378]))
            np.testing.assert_allclose(indicators[:, 1], 
                                    np.array([1077.48807619,  119.43464571]))
        finally:
            self.indicator.reset()

    def test_on_balance_volume(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator._on_balance_volume(market)
        obv1 = self.indicator.obv.copy()
        market[:, TestConfig.CLOSE_IX] = np.array([0.068, 3.8])
        self.indicator._on_balance_volume(market)
        try:
            np.testing.assert_allclose(obv1,
                                       np.array([11313.6248 ,  1254.06378]))
            np.testing.assert_allclose(self.indicator.obv,
                                       np.array([11313.6248, 0.]))
        finally:
            self.indicator.reset()

    def test_obv_ema(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator._on_balance_volume(market)
        self.indicator._obv_ema()
        try:
            np.testing.assert_allclose(self.indicator.obv_ema, 
                                    np.array([1077.48807619,  119.43464571]))
        finally:
            self.indicator.reset()

    def test_reset(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator.interpret(market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.last_close, np.array([0.0, 0.0]))
        np.testing.assert_allclose(self.indicator.obv, np.array([0.0, 0.0]))
        np.testing.assert_allclose(self.indicator.obv_ema, np.array([0.0, 0.0]))


class TestAwesomeOscillator:
    indicator = AwesomeOscillator(TestConfig)

    def test_interpret(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator.interpret(market)
        try:
            np.testing.assert_allclose(self.indicator.ema5, 
                                       np.array([0.013608, 0.785]))
            np.testing.assert_allclose(self.indicator.ema34, 
                                       np.array([0.00200117647058, 0.115441176470]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator.interpret(market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators, 
                                       np.array([[0.0116068235294], [0.66955882352]]))
        finally:
            self.indicator.reset()

    def test_reset(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.indicator.interpret(market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.ema34, np.array([0.0, 0.0]))
        np.testing.assert_allclose(self.indicator.ema5, np.array([0.0, 0.0]))


class TestSimpleMovingAverage:
    indicator = SimpleMovingAverage(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                        [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
    sma30 = np.array([0.002266666666, 0.13046666666])
    sma60 = np.array([0.001133333333, 0.06523333333])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.hist[:, -1], 
                                       np.array([6.80000000e-02, 3.91400000e+00]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], self.sma30)
            np.testing.assert_allclose(indicators[:, 1], self.sma60)
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.hist, 
                                   np.zeros((TestConfig.N_ASSETS, 60)))


class TestExponentialMovingAverage:
    indicator = ExponentialMovingAverage(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
    ema30 = np.array([0.002266666666, 0.13046666666])
    ema60 = np.array([0.001133333333, 0.06523333333])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.ema30, self.ema30)
            np.testing.assert_allclose(self.indicator.ema60, self.ema60)
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], self.ema30)
            np.testing.assert_allclose(indicators[:, 1], self.ema60)
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.ema30, 
                                   np.zeros(TestConfig.N_ASSETS))
        np.testing.assert_allclose(self.indicator.ema60, 
                                   np.zeros(TestConfig.N_ASSETS))


class TestAverageDirectionalIndex:
    indicator = AverageDirectionalIndex(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.market, self.market)
            np.testing.assert_allclose(self.indicator.TR_hist[:, -1], 
                                       np.array([0.06808, 3.927]))
            np.testing.assert_allclose(self.indicator.ema_DM_pos, 
                                       np.array([0.00453866666, 0.2618]))
            np.testing.assert_allclose(self.indicator.ema_DM_neg, np.array([0.0, 0.0]))
            np.testing.assert_allclose(self.indicator.prev_adx, np.array([0.0, 0.0]))
            np.testing.assert_allclose(self.indicator.adx, 
                                       np.array([20./3, 20./3]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], np.array([20./3, 20./3]))
            np.testing.assert_allclose(indicators[:, 1], np.array([0.0, 0.0]))
            np.testing.assert_allclose(indicators[:, 2], np.array([0.0, 0.0]))
            np.testing.assert_allclose(indicators[:, 3], np.array([0.00453866666, 0.2618]))
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.market, np.zeros((2, 7)))
        np.testing.assert_allclose(self.indicator.TR_hist, np.zeros((2, 14)))
        np.testing.assert_allclose(self.indicator.ema_DM_pos, np.zeros(2))
        np.testing.assert_allclose(self.indicator.ema_DM_neg, np.zeros(2))
        np.testing.assert_allclose(self.indicator.prev_adx, np.zeros(2))
        np.testing.assert_allclose(self.indicator.adx, np.zeros(2))


class TestAroon:
    indicator = Aroon(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
    aroon_up = np.array([100.0, 100.0])
    aroon_down = np.array([100.0, 100.0])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.high[:, -1], np.array([0.06808, 3.927]))
            np.testing.assert_allclose(self.indicator.low[:, -1], np.array([0.06799, 3.908]))
            np.testing.assert_allclose(self.indicator.aroon_up, self.aroon_up)
            np.testing.assert_allclose(self.indicator.aroon_down, self.aroon_down)
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], self.aroon_up)
            np.testing.assert_allclose(indicators[:, 1], self.aroon_down)
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.high, np.full((2, 26), np.nan))
        np.testing.assert_allclose(self.indicator.low, np.full((2, 26), np.nan))
        np.testing.assert_allclose(self.indicator.aroon_up, np.zeros(2))
        np.testing.assert_allclose(self.indicator.aroon_down, np.zeros(2))


class TestMovingAverageConvergenceDivergence:
    indicator = MovingAverageConvergenceDivergence(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
    macd = np.array([0.00305128205128, 0.17562820512])
    ema_12 = np.array([0.00566666666667, 0.32616666667])
    ema_26 = np.array([0.00261538461538, 0.15053846154])
    ema_macd = np.array([0.00033903133903, 0.01951424501])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.macd, self.macd)
            np.testing.assert_allclose(self.indicator.ema_12, self.ema_12)
            np.testing.assert_allclose(self.indicator.ema_26, self.ema_26)
            np.testing.assert_allclose(self.indicator.ema_macd, self.ema_macd)
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], self.macd)
            np.testing.assert_allclose(indicators[:, 1], self.ema_macd)
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.macd, np.zeros(2))
        np.testing.assert_allclose(self.indicator.ema_12, np.zeros(2))
        np.testing.assert_allclose(self.indicator.ema_26, np.zeros(2))
        np.testing.assert_allclose(self.indicator.ema_macd, np.zeros(2))


class TestRelativeStrengthIndex:
    indicator = RelativeStrengthIndex(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.latest_close, np.array([0.068, 3.914]))
            np.testing.assert_allclose(self.indicator.ema_up_change,
                                       np.array([0.0048572357142857145, 0.27957152142857145]))
            np.testing.assert_allclose(self.indicator.ema_down_change, 
                                       np.array([9.285714e-08, 9.285714e-08]))
            np.testing.assert_allclose(self.indicator.rsi, np.array([99.998088, 99.999967]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators, np.array([[99.99808831], [99.99996679]]))
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.latest_close, np.zeros(2))
        np.testing.assert_allclose(self.indicator.ema_up_change, np.zeros(2) + 1e-7)
        np.testing.assert_allclose(self.indicator.ema_down_change, np.zeros(2) + 1e-7)
        np.testing.assert_allclose(self.indicator.rsi, np.zeros(2))


class TestBollingerBands:
    indicator = BollingerBands(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.tp_hist[:, -1], 
                                       np.array([0.0680233333333, 3.916333333333]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], 
                                       np.array([0.0330518502469, 1.90290678959]))
            np.testing.assert_allclose(indicators[:, 1], 
                                       np.array([-0.0262495169136, -1.51127345626]))
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.tp_hist, np.zeros((2, 20)))


class TestStochasticOscillator:
    indicator = StochasticOscillator(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            assert self.indicator.counter == 1
            np.testing.assert_allclose(self.indicator.indicator_hist[:, -1], 
                                       np.array([99.88249119, 99.66895849]))
            np.testing.assert_allclose(self.indicator.indicator, 
                                       np.array([99.88249119, 99.66895849]))
            np.testing.assert_allclose(self.indicator.low14[:, 0], 
                                       np.array([0.06799, 3.908]))
            np.testing.assert_allclose(self.indicator.high14[:, 0], 
                                       np.array([0.06808, 3.927]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators[:, 0], np.array([99.88249119, 99.66895849]))
            np.testing.assert_allclose(indicators[:, 1], np.array([33.29416373, 33.22298616]))
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        assert self.indicator.counter == 0
        np.testing.assert_allclose(self.indicator.indicator_hist, np.zeros((2, 3)))
        np.testing.assert_allclose(self.indicator.indicator, np.zeros(2))
        np.testing.assert_allclose(self.indicator.low14, np.zeros((2, 14)))
        np.testing.assert_allclose(self.indicator.high14, np.zeros((2, 14)))


class TestCommodityChannelIndex:
    indicator = CommodityChannelIndex(TestConfig)
    market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                        1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                       [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                        1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])

    def test_interpret(self):
        self.indicator.interpret(self.market)
        try:
            np.testing.assert_allclose(self.indicator.tp_hist[:, -1], 
                                       np.array([0.06802333, 3.91633333]))
            np.testing.assert_allclose(self.indicator.diff_hist[:, -1], 
                                       np.array([0.06462217, 3.72051667]))
            np.testing.assert_allclose(self.indicator.cci, 
                                       np.array([1333.33333333, 1333.33333333]))
        finally:
            self.indicator.reset()

    def test_get_indicators(self):
        self.indicator.interpret(self.market)
        indicators = self.indicator.get_indicators()
        try:
            np.testing.assert_allclose(indicators, 
                                       np.array([[1333.33333333], [1333.33333333]]))
        finally:
            self.indicator.reset()

    def test_reset(self):
        self.indicator.interpret(self.market)
        self.indicator.reset()
        np.testing.assert_allclose(self.indicator.tp_hist, np.zeros((2, 20)))
        np.testing.assert_allclose(self.indicator.diff_hist, np.zeros((2, 20)))
        np.testing.assert_allclose(self.indicator.cci, np.zeros(2))
