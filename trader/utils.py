import numpy as np


def shift_window(window, new):
    shifted = np.zeros_like(window)
    shifted[:, :-1] = window[:, 1:]
    shifted[:, -1] = new
    return shifted