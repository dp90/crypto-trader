import numpy as np


def shift_window(window, new):
    shifted = np.zeros_like(window)
    shifted[:, :-1] = window[:, 1:]
    shifted[:, -1] = new
    return shifted


def get_scale_config(trading_config, data: np.ndarray) -> dict:
    """
    data : np.ndarray
        shape (n_assets, n_time_steps, n_indicators)
    """
    sc = {}
    for asset in range(trading_config.N_ASSETS):
        for indicator in range(trading_config.N_INDICATORS + 3):
            mean = data[asset, 50:, indicator].mean()
            sigma = (data[asset, 50:, indicator].max() -\
                     data[asset, 50:, indicator].min()) / 2
            sc[asset*(trading_config.N_INDICATORS + 3) + indicator] = \
                (mean, sigma)
    return sc
