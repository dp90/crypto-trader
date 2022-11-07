import pytest
import numpy as np

from trader.data_loader import BinanceDataLoader
from trader.simulate import BinanceSimulator
from configs import RESOURCES_PATH, TestConfig as TC


class TestBinanceSimulator:
    data_loader = BinanceDataLoader(RESOURCES_PATH, TC)
    simulator = BinanceSimulator(data_loader, TC.INITIAL_PORTFOLIO)

    def test_is_valid(self):
        order = np.array([-0.5, -13.6, 14.1])
        assert self.simulator._is_valid(order)

    def test_is_valid_error_sum_not_zero(self):
        order = np.array([-0.5, -13.6, 15.1])
        with pytest.raises(ValueError):
            self.simulator._is_valid(order)

    def test_is_valid_error_neg_funds(self):
        order = np.array([-15, -1.6, 16.6])
        with pytest.raises(ValueError):
            self.simulator._is_valid(order)

    def test_get_market_data(self):
        data = self.simulator._get_market_data()
        ref_data = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                              1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                             [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                              1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        np.testing.assert_allclose(data, ref_data)
        self.simulator.data_loader.reset()

    def test_is_executable(self):
        pass

    def test_update_portfolio(self):
        pass

    def test_reset(self):
        pass

    def test_execute(self):
        pass