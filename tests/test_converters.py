import numpy as np
from configs import TestConfig

from trader.converters import MarketOrderConverter, MarketInterpreter
from trader.indicators import CommodityChannelIndex, OnBalanceVolume
from trader.validators import BookKeeper


class TestMarketOrderConverter:
    book_keeper = BookKeeper(TestConfig)
    converter = MarketOrderConverter(book_keeper)

    def test_convert(self):
        action = np.array([0.25, 0.3, 0.45])
        order = self.converter.convert(action)
        ref_order = np.array([-4.131, 78.50970018, -0.30848408])
        np.testing.assert_allclose(order, ref_order)

    def test_normalize(self):
        action = np.array([-0.2, 0.8, 1.4])
        np.testing.assert_allclose(self.converter._normalize(action),
                                   np.array([0.0, 0.38461538, 0.61538462]))


class TestMarketInterpreter:
    technical_indicators = [CommodityChannelIndex(TestConfig),
                            OnBalanceVolume(TestConfig)]
    market_interpreter = MarketInterpreter(technical_indicators)
    
    def test_interpret(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        indicators = self.market_interpreter.interpret(market)
        try:
            np.testing.assert_allclose(indicators,
                                       np.array([[1333.33333333, 11313.6248, 1077.48807619],
                                                 [1333.33333333, 1254.06378, 119.43464571]]))
        finally:
            self.market_interpreter.reset()

    def test_reset(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        _ = self.market_interpreter.interpret(market)
        self.market_interpreter.reset()
        np.testing.assert_allclose(self.market_interpreter.indicators[0].tp_hist, np.zeros((2, 20)))
        np.testing.assert_allclose(self.market_interpreter.indicators[0].diff_hist, np.zeros((2, 20)))
        np.testing.assert_allclose(self.market_interpreter.indicators[0].cci, np.zeros(2))
        np.testing.assert_allclose(self.market_interpreter.indicators[1].last_close, np.zeros(2))
        np.testing.assert_allclose(self.market_interpreter.indicators[1].obv, np.zeros(2))
        np.testing.assert_allclose(self.market_interpreter.indicators[1].obv_ema, np.zeros(2))
