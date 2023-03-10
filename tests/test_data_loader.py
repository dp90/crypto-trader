import os
import pandas as pd
import numpy as np
import pytest

from tests.configs import RESOURCES_PATH, TestConfig
from trader.data_loader import BinanceDataLoader


class TestDataLoader:
    path = RESOURCES_PATH
    data_loader = BinanceDataLoader(path, TestConfig)
    assets = TestConfig.CURRENCIES

    def test_read_data(self):
        raw_path = os.path.join(self.path, 'raw')
        data = self.data_loader._read_data(raw_path, self.assets)
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
        raw_path = os.path.join(self.path, 'raw')
        raw = self.data_loader._read_data(raw_path, self.assets)
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
        ref_data = np.array([[[ 6.80400000e-02, 6.80800000e-02, 6.79900000e-02, 6.80000000e-02, 1.13136248e+04, 1.80000000e+01,
          1.00000000e+00,  1.13136248e+04,  1.07748808e+03, 1.16068235e-02,  2.26666667e-03,  1.13333333e-03,
          2.26666667e-03,  1.13333333e-03,  6.66666667e+00, 0.00000000e+00,  0.00000000e+00,  4.53866667e-03,
          1.00000000e+02,  1.00000000e+02,  3.05128205e-03, 3.39031339e-04,  9.99980883e+01, -8.79948598e+03,
          3.30518502e-02, -2.62495169e-02,  9.98824912e+01, 3.32941637e+01,  1.33333333e+03],
        [ 6.79900000e-02,  6.80000000e-02,  6.79600000e-02, 6.80000000e-02,  3.81227816e+03,  1.60000000e+01,
          2.00000000e+00,  1.13136248e+04,  2.05235825e+03, 2.05423758e-02,  4.53333333e-03,  2.26666667e-03,
          4.45777778e-03,  2.24777778e-03,  1.28825967e+01, 6.66666667e+00,  2.00000000e-06,  4.23608889e-03,
          9.60000000e+01,  1.00000000e+02,  5.73093360e-03, 9.38131590e-04,  9.99980883e+01, -4.98720782e+03,
          4.76035016e-02, -3.40025016e-02,  9.98824912e+01, 6.65883275e+01,  6.48459076e+02]],
       [[ 3.92500000e+00,  3.92700000e+00,  3.90800000e+00, 3.91400000e+00,  1.25406378e+03,  4.00000000e+00,
          1.00000000e+00,  1.25406378e+03,  1.19434646e+02, 6.69558824e-01,  1.30466667e-01,  6.52333333e-02,
          1.30466667e-01,  6.52333333e-02,  6.66666667e+00, 0.00000000e+00,  0.00000000e+00,  2.61800000e-01,
          1.00000000e+02,  1.00000000e+02,  1.75628205e-01, 1.95142450e-02,  9.99999668e+01, -4.62023498e+02,
          1.90290679e+00, -1.51127346e+00,  9.96689585e+01, 3.32229862e+01,  1.33333333e+03],
        [ 3.91900000e+00,  3.91900000e+00,  3.91900000e+00, 3.91900000e+00,  2.70583436e+02,  1.00000000e+00,
          2.00000000e+00,  1.52464722e+03,  2.53264414e+02, 1.18448945e+00,  2.61100000e-01,  1.30550000e-01,
          2.56751111e-01,  1.29462778e-01,  1.28888889e+01, 6.66666667e+00,  0.00000000e+00,  2.44346667e-01,
          9.60000000e+01,  9.60000000e+01,  3.30090155e-01, 5.40226794e-02,  9.99999668e+01, -4.62023498e+02,
          2.74236682e+00, -1.95883348e+00,  9.97962821e+01, 6.64884135e+01,  6.48887969e+02]]])
        np.testing.assert_allclose(data, ref_data)

    def test_next(self):
        market = self.data_loader.next()
        market_ref = np.array([
            [ 1.00000000e+02,  3.05128205e-03,  3.39031339e-04, 9.99980883e+01, -8.79948598e+03,  3.30518502e-02,
                        -2.62495169e-02,  9.98824912e+01,  3.32941637e+01, 1.33333333e+03],
            [ 1.00000000e+02,  1.75628205e-01,  1.95142450e-02, 9.99999668e+01, -4.62023498e+02,  1.90290679e+00,
                        -1.51127346e+00,  9.96689585e+01,  3.32229862e+01, 1.33333333e+03]])
        try:
            np.testing.assert_allclose(market[:, -10:], market_ref)
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
