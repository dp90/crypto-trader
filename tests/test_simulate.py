import pytest
import numpy as np

from trader.data_loader import BinanceDataLoader
from trader.simulate import MarketOrderBinanceSimulator, LimitOrderBinanceSimulator
from configs import RESOURCES_PATH, TestConfig as TC


class TestMarketOrderBinanceSimulator:
    data_loader = BinanceDataLoader(RESOURCES_PATH, TC)
    simulator = MarketOrderBinanceSimulator(data_loader, TC)

    def test_is_valid(self):
        order = np.array([-0.5, -13.6, 14.1])
        assert self.simulator._is_valid(order)

    def test_is_valid_error_neg_funds(self):
        order = np.array([-15, -1.6, 16.6])
        with pytest.raises(ValueError):
            self.simulator._is_valid(order)

    def test_get_market_data(self):
        data = self.simulator.get_market_data()
        ref_data = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                              1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                             [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                              1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        try:
            np.testing.assert_allclose(data[:, :7], ref_data)
        finally:
            self.simulator.data_loader.reset()

    def test_get_trades(self):
        order = np.array([-4.131, 78.50970018, -0.30848408])
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        trade = self.simulator._get_trades(order, market)
        ref_trade = np.array([-4.2031199, 78.50970018, -0.30848408])
        try:
            np.testing.assert_allclose(trade, ref_trade)
        finally:
            self.simulator.data_loader.reset()

    def test_update_portfolio(self):
        trades = np.array([-4.2031199, 78.50970018, -0.30848408])
        self.simulator._update_portfolio(trades)
        ref_portfolio = np.array([5.7968801, 103.50970018, 2.69151592])
        try:
            np.testing.assert_allclose(self.simulator.portfolio, ref_portfolio)
        finally:
            self.simulator.portfolio = TC.INITIAL_PORTFOLIO.copy()

    def test_reset(self):
        with pytest.raises(NotImplementedError):
            self.simulator.reset()

    def test_execute(self):
        order = np.array([-4.131, 78.50970018, -0.30848408])
        market_data, portfolio = self.simulator.execute(order)
        
        ref_market_data = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                                     1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                                   [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                                     1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        ref_portfolio = np.array([5.7968801, 103.50970018, 2.69151592])
        np.testing.assert_allclose(market_data[:, :7], ref_market_data)
        np.testing.assert_allclose(portfolio, ref_portfolio)


class TestLimitOrderBinanceSimulator:
    data_loader = BinanceDataLoader(RESOURCES_PATH, TC)
    simulator = LimitOrderBinanceSimulator(data_loader, TC)

    def test_get_trades(self):
        order = np.array([0., 0., 50., 10., 1., 10., 0., 0.])
        market = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                            1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                           [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                            1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        self.simulator._get_trades(order, market)