import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

%matplotlib notebook

def OU(T, mu, sigma, dt):
    """
    Simulate an Ornstein-Uhlenbeck process
    
    Parameters
    ----------
    T : int
        Length of the time series
    mu : float
        Mean reversion level
    sigma : float
        Volatility
    dt : float
        Time step
    
    Returns
    -------
    pd.Series
        Time series of the Ornstein-Uhlenbeck process
    """
    x = np.zeros(T)
    x[0] = 0
    for t in range(1, T):
        x[t] = x[t-1] + mu * dt + sigma * np.random.normal(0, np.sqrt(dt))
    return pd.Series(x)

def plot_OU(T, mu, sigma, dt):
    x = OU(T, mu, sigma, dt)
    plt.plot(x)
    plt.title('Ornstein-Uhlenbeck process')
    plt.show()



    # come back to this later