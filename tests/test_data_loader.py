import pandas as pd
import numpy as np
import os
import pytest
from trader.configs import TradingConfig

from trader.data_loader import BinanceDataLoader

class TestConfig(TradingConfig):
    START_CAPITAL = 1000
    N_ASSETS = 5
    VARIABLES = ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'N_TRADES']
    N_VARIABLES = len(VARIABLES)
    CURRENCY_ICS = [1, 2, 3]
    CURRENCIES = ['ADA', 'ATOM']


class TestDataLoader:
    path = os.path.join(os.getcwd(), 'tests', 'resources')
    data_loader = BinanceDataLoader(path, TestConfig)

    def test_read_data(self):
        data = self.data_loader._read_data(self.path)
        ref_df0 = pd.DataFrame({'TIME': {0: '2019-05-01 00:05:00', 1: '2019-05-01 00:10:00'}, 
                                'OPEN': {0: 0.06804, 1: 0.06799}, 
                                'HIGH': {0: 0.06808, 1: 0.068}, 
                                'LOW': {0: 0.06799, 1: 0.06796}, 
                                'CLOSE': {0: 0.068, 1: 0.068}, 
                                'VOLUME': {0: 11313.624827, 1: 3812.27816}, 
                                'NUMBER_OF_TRADES': {0: 18, 1: 16}})
        ref_df1 = pd.DataFrame({'TIME': {0: '2019-05-01 00:05:00', 1: '2019-05-01 00:10:00'}, 
                                'OPEN': {0: 3.925, 1: 3.919}, 
                                'HIGH': {0: 3.927, 1: 3.919}, 
                                'LOW': {0: 3.908, 1: 3.919}, 
                                'CLOSE': {0: 3.914, 1: 3.919}, 
                                'VOLUME': {0: 1254.06378, 1: 270.583436}, 
                                'NUMBER_OF_TRADES': {0: 4, 1: 1}})
        pd.testing.assert_frame_equal(ref_df0, data[0])
        pd.testing.assert_frame_equal(ref_df1, data[1])

    def test_convert_data(self):
        raw = self.data_loader._read_data(self.path)
        data = self.data_loader._convert_data(raw)
        ref_data = np.array([[[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                               1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                              [6.79900000e-02, 6.80000000e-02, 6.79600000e-02, 6.80000000e-02,
                               3.81227816e+03, 1.60000000e+01, 2.00000000e+00]],

                             [[3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                               1.25406378e+03, 4.00000000e+00, 1.00000000e+00],
                              [3.91900000e+00, 3.91900000e+00, 3.91900000e+00, 3.91900000e+00,
                               2.70583436e+02, 1.00000000e+00, 2.00000000e+00]]])
        np.testing.assert_allclose(data, ref_data)

    def test_load_data(self):
        data = self.data_loader._load_data(self.path)
        ref_data = np.array([[[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                               1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                              [6.79900000e-02, 6.80000000e-02, 6.79600000e-02, 6.80000000e-02,
                               3.81227816e+03, 1.60000000e+01, 2.00000000e+00]],

                             [[3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                               1.25406378e+03, 4.00000000e+00, 1.00000000e+00],
                              [3.91900000e+00, 3.91900000e+00, 3.91900000e+00, 3.91900000e+00,
                               2.70583436e+02, 1.00000000e+00, 2.00000000e+00]]])
        np.testing.assert_allclose(data, ref_data)

    def test_next(self):
        market = self.data_loader.next()
        market_ref = np.array([[6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02,
                                1.13136248e+04, 1.80000000e+01, 1.00000000e+00],
                               [3.92500000e+00, 3.92700000e+00, 3.90800000e+00, 3.91400000e+00,
                                1.25406378e+03, 4.00000000e+00, 1.00000000e+00]])
        try:
            np.testing.assert_allclose(market, market_ref)
            assert self.data_loader.index == 1
        finally:
            self.data_loader.reset()

    def test_next_error(self):
        _ = self.data_loader.next()
        _ = self.data_loader.next()
        with pytest.raises(IndexError):
            _ = self.data_loader.next()
        self.data_loader.reset()
        
    def test_reset(self):
        _ = self.data_loader.next()
        self.data_loader.reset()
        assert self.data_loader.index == 0
