import numpy as np
import matplotlib.pyplot as plt


def plot_running_average(x, window=100):
    N = len(x)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(x[max(0, t - window): (t + 1)])
    plt.figure()
    plt.plot(running_avg)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards (running average)")
    plt.show()
    return
