import pytest
import numpy as np

from configs import TestConfig as TC
from trader.validators import BookKeeper


class TestBookKeeper:
    book_keeper = BookKeeper(TC.INITIAL_PORTFOLIO)

    def test_update(self):
        new_portfolio = np.array([8, 23, 2])
        self.book_keeper.update(new_portfolio)
        np.testing.assert_array_equal(self.book_keeper.portfolio, 
                                      new_portfolio)
        self.book_keeper.portfolio = TC.INITIAL_PORTFOLIO
    
    def test_update_error_wrong_size(self):
        new_portfolio = np.array([24, 36])
        with pytest.raises(ValueError):
            self.book_keeper.update(new_portfolio)
        self.book_keeper.portfolio = TC.INITIAL_PORTFOLIO

    def test_is_valid(self):
        portfolio_weights = np.array([0.25, 0.3, 0.45])
        assert self.book_keeper.is_valid(portfolio_weights)

    def test_is_valid_error_shorting(self):
        portfolio_weights = np.array([0.85, 0.6, -0.45])
        with pytest.raises(ValueError):
            self.book_keeper.is_valid(portfolio_weights)

    def test_is_valid_error_sum_not_one(self):
        portfolio_weights = np.array([0.85, 0.6, 0.45])
        with pytest.raises(ValueError):
            self.book_keeper.is_valid(portfolio_weights)
        