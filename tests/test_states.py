import numpy as np

from configs import RESOURCES_PATH, TestConfig as TC
from trader.states import create_hist_state_processor


class TestStateProcessor:
    state_processor = create_hist_state_processor(TC, RESOURCES_PATH)

    def test_get_initial_state(self):
        state = self.state_processor.get_initial_state()
        ref_state = np.array([ 1.13136248e+04,  1.07748808e+03,  1.16068235e-02,  2.26666667e-03,
                               1.13333333e-03,  2.26666667e-03,  1.13333333e-03,  6.66666667e+00,
                               0.00000000e+00,  0.00000000e+00,  4.53866667e-03,  1.00000000e+02,
                               1.00000000e+02,  3.05128205e-03,  3.39031339e-04,  9.99980883e+01,
                               -8.79948598e+03,  3.30518502e-02, -2.62495169e-02,  9.98824912e+01,
                               3.32941637e+01,  1.33333333e+03,  1.25406378e+03,  1.19434646e+02,
                               6.69558824e-01,  1.30466667e-01,  6.52333333e-02,  1.30466667e-01,
                               6.52333333e-02,  6.66666667e+00,  0.00000000e+00,  0.00000000e+00,
                               2.61800000e-01,  1.00000000e+02,  1.00000000e+02,  1.75628205e-01,
                               1.95142450e-02,  9.99999668e+01, -4.62023498e+02,  1.90290679e+00,
                               -1.51127346e+00,  9.96689585e+01,  3.32229862e+01,  1.33333333e+03,
                               1.00000000e+01,  2.50000000e+01,  3.00000000e+00])
        try:
            np.testing.assert_allclose(state, ref_state)
        finally:
            self.state_processor.reset()
    
    def test_update_state(self):
        state = self.state_processor.get_initial_state()
        action = np.ones(TC.N_ASSETS + 1) * (1 / (TC.N_ASSETS + 1))
        state_ = self.state_processor.update_state(state, action)
        ref_state_ = np.array([ 1.13136248e+04,  2.05235825e+03,  2.05423758e-02,  4.53333333e-03,
                                2.26666667e-03,  4.45777778e-03,  2.24777778e-03,  1.28825967e+01,
                                6.66666667e+00,  2.00000000e-06,  4.23608889e-03,  9.60000000e+01,
                                1.00000000e+02,  5.73093360e-03,  9.38131590e-04,  9.99980883e+01,
                               -4.98720782e+03,  4.76035016e-02, -3.40025016e-02,  9.98824912e+01,
                                6.65883275e+01,  6.48459076e+02,  1.52464722e+03,  2.53264414e+02,
                                1.18448945e+00,  2.61100000e-01,  1.30550000e-01,  2.56751111e-01,
                                1.29462778e-01,  1.28888889e+01,  6.66666667e+00,  0.00000000e+00,
                                2.44346667e-01,  9.60000000e+01,  9.60000000e+01,  3.30090155e-01,
                                5.40226794e-02,  9.99999668e+01, -4.62023498e+02,  2.74236682e+00,
                               -1.95883348e+00,  9.97962821e+01,  6.64884135e+01,  6.48887969e+02,
                                7.70938789e+00,  1.14911765e+02,  1.99642310e+00])
        try:
            np.testing.assert_allclose(state_, ref_state_)
        finally:
            self.state_processor.reset()

    def test_reset(self):
        _ = self.state_processor.get_initial_state()
        self.state_processor.reset()
        np.testing.assert_allclose(
            self.state_processor.book_keeper.portfolio,
            np.array([10., 25., 3.])
        )        
        np.testing.assert_allclose(
            self.state_processor.book_keeper.exchange_rate,
            np.array([1., 0.06804, 3.925])
        )
        np.testing.assert_allclose(
            self.state_processor.binance.portfolio,
            np.array([10., 25., 3.])
        )
        assert self.state_processor.binance.data_loader.index == 0
        assert not self.state_processor.binance.data_loader.is_final_time_step
