import numpy as np

from trader.converters import ActionConverter
from trader.validators import BookKeeper
from configs import TestConfig


class TestActionConverter:
    book_keeper = BookKeeper(TestConfig.INITIAL_PORTFOLIO)
    converter = ActionConverter(book_keeper)

    def test_convert(self):
        action = np.array([0.25, 0.3, 0.45])
        order = self.converter.convert(action)
        ref_order = np.array([-0.5, -13.6, 14.1])
        np.testing.assert_allclose(order, ref_order)
