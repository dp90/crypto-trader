from abc import abstractmethod
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from abc import ABC
import torch.nn as nn
import scipy.signal

from configs import DirectoryConfig as DIR, Mode


class IDataLoader(ABC):
    """
    Interface to load scenario data. If historic data is available, it is assumed to be one trajectory and is prepended to the
    scenario data. 
    If more data sources are added, consider a DataWarehouse class that reads data from multiple DataLoaders.
    """
    def load_scenario_data(self, mode: Mode):
        data = pd.read_csv(os.path.join(DIR.DATA, mode.filename))
        return self._convert_scenario_data(data, mode)

    def load_historic_data(self):
        data = pd.read_csv(os.path.join(DIR.DATA, 'historic_data.csv'))
        return self._convert_historic_data(data)

    @abstractmethod
    def _convert_scenario_data(self, data: pd.DataFrame, mode: Mode):
        raise NotImplementedError

    @abstractmethod
    def _convert_historic_data(self, data):
        raise NotImplementedError


class Scaler(object):
    """
    Scales state variables as specified in scaling configuration. State is assumed to be of shape M x N. If 
    M > 1, the state is assumed to be a batch of states. Scaling config is a dictionary with keys denoting the 
    index of the state variable, and values a tuple of the mean and half the distance between min and max values 
    (denoted sigma).
    """

    def __init__(self, scaling_config: dict):
        self.scaling_config = scaling_config

    def scale(self, array: np.ndarray):
        data = array.copy()
        if data.ndim == 1:
            data = data[None, :]
        for index, (mean, sigma) in self.scaling_config.items():
            data[:, index] = (data[:, index] - mean) / sigma
        return data

    def unscale(self, array: np.ndarray):
        data = array.copy()
        if data.ndim == 1:
            data = data[None, :]
        for index, (mean, sigma) in self.scaling_config.items():
            data[:, index] = data[:, index] * sigma + mean
        return data
    

def mlp(sizes: List[int], activation: nn.Module, output_activation: nn.Module):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
