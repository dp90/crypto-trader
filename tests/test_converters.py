import numpy as np

from trader.converters import ActionConverter, MarketInterpreter
from trader.validators import BookKeeper
from configs import TestConfig


class TestActionConverter:
    book_keeper = BookKeeper(TestConfig.INITIAL_PORTFOLIO.copy(),
                             TestConfig.INITIAL_EXCHANGE_RATE.copy())
    converter = ActionConverter(book_keeper)

    def test_convert(self):
        action = np.array([0.25, 0.3, 0.45])
        order = self.converter.convert(action)
        ref_order = np.array([-4.131, 78.50970018, -0.30848408])
        np.testing.assert_allclose(order, ref_order)


class TestMarketInterpreter:
    
    market_interpreter = MarketInterpreter(TestConfig, )
    
    def test_on_balance_volume(self):
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.market_interpreter._on_balance_volume(market)
        obv1 = self.market_interpreter.obv.copy()
        self.market_interpreter.market_data = market.copy()
        market[:, TestConfig.CLOSE_IX] = np.array([0.068, 3.8])
        self.market_interpreter._on_balance_volume(market)
        try:
            np.testing.assert_allclose(obv1,
                                       np.array([11313.6248 ,  1254.06378]))
            np.testing.assert_allclose(self.market_interpreter.obv,
                                       np.array([11313.6248, 0.]))
        finally:
            self.market_interpreter.reset()
