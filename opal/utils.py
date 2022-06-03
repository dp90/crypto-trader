import pandas as pd
import numpy as np

from configs import IMode, Mode
from ofrl.utils import IDataLoader


class DataLoader(IDataLoader):
    def _convert_scenario_data(self, data: pd.DataFrame, mode: Mode):
        data.drop(columns=['Scenario', 'Year', 'Month'], inplace=True)

        # Reshape to 3D matrix of dims [num_years x num_scenarios x num_assets + 1] (+1 for inflation time series)
        array = data.values.reshape(41, mode.n_scenarios, -1, order="F")
        return array

    def _convert_historic_data(self, data):
        data.drop(columns=['Date'], inplace=True)
        return data.values
        
    def _combine_data_sources(self, data: np.ndarray, historic_data: np.ndarray):
        raise NotImplementedError


class EMode(IMode):
    @classmethod  # Ordering of decorators is important
    @property
    def TRAIN(cls):
        return Mode(name="train", filename="Annual US scenarios - Train.csv", n_scenarios=10_500)
    
    @classmethod  # Ordering of decorators is important
    @property
    def TEST(cls):
        return Mode(name="test", filename="Annual US scenarios - Test.csv", n_scenarios=2_250)
    
    @classmethod  # Ordering of decorators is important
    @property
    def DEVELOP(cls):
        return Mode(name="develop", filename="Annual US scenarios - Develop.csv", n_scenarios=2_250)


def update_old_weights(returns: np.ndarray, old_weights: np.ndarray) -> np.ndarray:
    """
    Update the old weights based on the returns of the previous time step.
    """
    weights_numerator = (1 + returns) * old_weights
    weights_denominator = np.tile(np.sum(weights_numerator, axis=1)[:, np.newaxis], (1, 7))
    previous_weights = np.divide(weights_numerator, weights_denominator)
    return previous_weights


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    new_weights_denominator = np.tile(np.sum(weights, axis=1)[:, np.newaxis], (1, 7))
    new_weights = np.divide(weights, new_weights_denominator)
    return new_weights


def get_transaction_costs(new_weights, previous_weights, previous_portfolio_value, transaction_costs):
    return np.multiply(previous_portfolio_value, transaction_costs * np.sum(np.abs(new_weights - previous_weights), axis = 1)[:, np.newaxis])


def get_portfolio_returns(new_returns, new_weights):
    new_returns_numerator = (1 + np.sum(np.multiply(new_returns[:, 0:7], new_weights), axis=1))[:, np.newaxis]
    new_returns_denominator = (1 + new_returns[:, -1])[:, np.newaxis]
    return np.divide(new_returns_numerator, new_returns_denominator)


def update_portfolio_value(new_returns, new_weights, previous_portfolio_value):
    net_returns = get_portfolio_returns(new_returns, new_weights)
    new_portfolio_value = np.multiply(previous_portfolio_value, net_returns)
    return new_portfolio_value
