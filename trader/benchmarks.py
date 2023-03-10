import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage, sample_cov
from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR
from pypfopt import HRPOpt


class MvoAgent:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.c = self.data_loader.config
        self.start_ix = data_loader.config.DATA_START_INDEX - 289
        self.final_ix = data_loader.config.DATA_FINAL_INDEX - 289
    
    def next_mvo_weights(self) -> np.ndarray:
        returns_df = self._get_returns_df()
        return self._mvo_weights(returns_df)
    
    def next_mcvar_weights(self) -> np.ndarray:
        returns_df = self._get_returns_df()
        return self._mcvar_weights(returns_df)
    
    def next_hrp_weights(self) -> np.ndarray:
        returns_df = self._get_returns_df()
        returns_df.loc[0, 'USDT'] = 1e-7
        hrp = HRPOpt(returns_df)
        hrp.optimize()
        return hrp.weights

    def _get_returns_df(self) -> pd.DataFrame:
        returns_data = self.data_loader.data[:, self.start_ix: self.final_ix, self.c.RETURNS_IX].T
        returns_data = np.hstack((returns_data, np.zeros((len(returns_data), 1))))
        df = pd.DataFrame(returns_data, columns=self.c.CURRENCIES + ['USDT'])
        self.start_ix += 1
        self.final_ix += 1
        return df

    def _mvo_weights(self, return_data: pd.DataFrame) -> np.ndarray:
        mu = mean_historical_return(return_data, returns_data=True)
        S = CovarianceShrinkage(return_data, returns_data=True).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        return ef.weights

    def _mcvar_weights(self, return_data: pd.DataFrame) -> np.ndarray:
        mu = mean_historical_return(return_data, returns_data=True)
        S = sample_cov(return_data, returns_data=True)
        ef_cvar = EfficientCVaR(mu, S)
        ef_cvar.min_cvar()
        return ef_cvar.weights
