import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def plot_moving_average(values, window):
    ma = moving_average(values, window)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title("Learning Curve")
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Average of Epoch Rewards')
    ax.plot(ma, color='black')
    plt.show()
