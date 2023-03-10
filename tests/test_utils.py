import numpy as np

from trader.utils import shift_window


class TestUtils:
    array = np.array([[1, 2], [3, 4]])

    def test_shift_window(self):
        new = np.array([5, 6])
        shifted = shift_window(self.array, new)
        shifted_ref = np.array([[2, 5], [4, 6]])
        np.testing.assert_allclose(shifted, shifted_ref)
