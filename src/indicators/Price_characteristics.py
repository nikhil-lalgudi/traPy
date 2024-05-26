import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

def average_true_range(df, n):
    df['HL'] = df['High'] - df['Low']
    df['HC'] = np.abs(df['High'] - df['Close'].shift())
    df['LC'] = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n, min_periods=1).mean()
    return df['ATR']

def balance_of_power(df):
    df['BOP'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    return df['BOP']

def bull_bear_power(df, n):
    df['EMA'] = df['Close'].ewm(span=n, adjust=False).mean()
    df['Bull Power'] = df['High'] - df['EMA']
    df['Bear Power'] = df['Low'] - df['EMA']
    return df['Bull Power'], df['Bear Power']

def choppiness_index(df, n):
    df['TR'] = df[['High', 'Low', 'Close']].diff().abs().sum(axis=1)
    ATR_sum = df['TR'].rolling(window=n).sum()
    CHOP = 100 * np.log10(ATR_sum / (df['High'].rolling(window=n).max() - df['Low'].rolling(window=n).min())) / np.log10(n)
    return CHOP

def dominant_cycle_periods(df):
    close_diff = df['Close'].diff().dropna()
    peaks, _ = find_peaks(close_diff)
    peak_distances = np.diff(peaks)
    if len(peak_distances) > 0:
        dominant_period = np.median(peak_distances)
    else:
        dominant_period = np.nan
    return pd.Series(dominant_period, index=df.index)

def historical_volatility(df, n):
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    HV = df['Log_Return'].rolling(window=n).std() * np.sqrt(252)
    return HV

def hurst_exponent(df, n):
    lags = range(2, n)
    tau = [np.sqrt(np.std(df['Close'].diff(lag))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def momentum_oscillator(df, n):
    MO = df['Close'] - df['Close'].shift(n)
    return MO

def normalized_average_true_range(df, n):
    df['ATR'] = average_true_range(df, n)
    NATR = (df['ATR'] / df['Close']) * 100
    return NATR

def price_momentum_oscillator(df):
    df['ROC1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
    df['ROC2'] = df['ROC1'].ewm(span=35, adjust=False).mean()
    df['PMO'] = df['ROC2'].ewm(span=20, adjust=False).mean()
    return df['PMO']

def price_relative_strength(df, n):
    df['Price_Relative'] = df['Close'] / df['Close'].shift(n)
    return df['Price_Relative']

def rate_of_change(df, n):
    ROC = df['Close'].pct_change(periods=n) * 100
    return ROC

def rescaled_range_analysis(df, n):
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    rolling_window = df['Log_Return'].rolling(window=n)
    def rs(series):
        mean_adj = series - series.mean()
        cumulative_dev = mean_adj.cumsum()
        range_dev = cumulative_dev.max() - cumulative_dev.min()
        std_dev = series.std()
        return range_dev / std_dev if std_dev != 0 else 0
    R_S = rolling_window.apply(rs)
    return R_S

def true_range(df):
    df['HL'] = df['High'] - df['Low']
    df['HC'] = np.abs(df['High'] - df['Close'].shift())
    df['LC'] = np.abs(df['Low'] - df['Close'].shift())
    TR = df[['HL', 'HC', 'LC']].max(axis=1)
    return TR

def true_strength_index(df, r, s):
    diff = df['Close'].diff(1)
    abs_diff = abs(diff)
    double_smoothed = diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    double_abs_smoothed = abs_diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    TSI = 100 * (double_smoothed / double_abs_smoothed)
    return TSI

def ulcer_index(df, n):
    df['Drawdown'] = ((df['Close'] - df['Close'].rolling(window=n).max()) / df['Close'].rolling(window=n).max()) * 100
    UI = np.sqrt((df['Drawdown']**2).rolling(window=n).mean())
    return UI

## Numerical Analysis Functions

def beta_coefficient(x, y):
    x = np.array(x)
    y = np.array(y)
    covariance = np.cov(x, y, bias=True)[0, 1]
    variance_x = np.var(x, ddof=1)
    beta = covariance / variance_x
    return beta

def correlation_coefficient(x, y):
    x = np.array(x)
    y = np.array(y)
    corr_coef = np.corrcoef(x, y)[0, 1]
    return corr_coef

def linear_regression(x, y):
    slope, intercept, _, _, _ = linregress(x, y)
    return slope, intercept

def mean_absolute_deviation(data):
    data = np.array(data)
    mean = np.mean(data)
    mad = np.mean(np.abs(data - mean))
    return mad

def mean_absolute_percentage_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape

def mean_square_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mse = np.mean((actual - predicted) ** 2)
    return mse

def r_squared(x, y):
    slope, intercept = linear_regression(x, y)
    x = np.array(x)
    y = np.array(y)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def standard_deviation(data):
    data = np.array(data)
    std_dev = np.std(data, ddof=1)
    return std_dev

def z_score(data, value):
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    z_score = (value - mean) / std_dev
    return z_score