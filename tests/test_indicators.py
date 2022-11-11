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
            np.testing.assert_allclose(indicators[0], 
                                    np.array([11313.6248 ,  1254.06378]))
            np.testing.assert_allclose(indicators[1], 
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
            np.testing.assert_allclose(indicators[0], 
                                       np.array([0.0116068235294, 0.66955882352]))
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

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestExponentialMovingAverage:
    indicator = ExponentialMovingAverage(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestAverageDirectionalIndex:
    indicator = AverageDirectionalIndex(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestAroon:
    indicator = Aroon(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestMovingAverageConvergenceDivergence:
    indicator = MovingAverageConvergenceDivergence(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestRelativeStrengthIndex:
    indicator = RelativeStrengthIndex(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestBollingerBands:
    indicator = BollingerBands(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestStochasticOscillator:
    indicator = StochasticOscillator(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass


class TestCommodityChannelIndex:
    indicator = CommodityChannelIndex(TestConfig)

    def test_interpret(self):
        pass

    def test_get_indicators(self):
        pass

    def test_reset(self):
        pass
