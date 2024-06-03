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
    df['Drawdown'] = ((df['Close'] - df['Close'].rolling(window=n).max())
                       / df['Close'].rolling(window=n).max()) * 100
    UI = np.sqrt((df['Drawdown']**2).rolling(window=n).mean())
    return UI

def moving_average_crossover(data, short_window, long_window):
    short_ma = data['Close'].rolling(window=short_window).mean()
    long_ma = data['Close'].rolling(window=long_window).mean()
    
    crossover_points = []
    
    for i in range(1, len(data)):
        if short_ma[i] > long_ma[i] and short_ma[i - 1] <= long_ma[i - 1]:
            crossover_points.append((data.index[i], 'bullish'))
        elif short_ma[i] < long_ma[i] and short_ma[i - 1] >= long_ma[i - 1]:
            crossover_points.append((data.index[i], 'bearish'))
    
    return crossover_points

def plot_moving_average_crossover(data, short_window, long_window):
    crossover_points = moving_average_crossover(data, short_window, long_window)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    short_ma = data['Close'].rolling(window=short_window).mean()
    long_ma = data['Close'].rolling(window=long_window).mean()
    plt.plot(data.index, short_ma, label='{}-day MA'.format(short_window), color='orange')
    plt.plot(data.index, long_ma, label='{}-day MA'.format(long_window), color='green')

    for date, crossover_type in crossover_points:
        color = 'green' if crossover_type == 'bullish' else 'red'
        plt.scatter(date, data.loc[date, 'Close'], color=color, s=50, marker='o', label=crossover_type)

    plt.title('Moving Average Cross-Overs')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_cci(df, period=20):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['sma_tp'] = df['typical_price'].rolling(window=period).mean()
    df['mean_deviation'] = df['typical_price'].rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci'] = (df['typical_price'] - df['sma_tp']) / (0.015 * df['mean_deviation'])
    return df

def plot_cci(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(df['date'], df['cci'], label='CCI', color='blue')
    ax2.axhline(100, color='red', linestyle='--')
    ax2.axhline(-100, color='red', linestyle='--')
    ax2.set_title('Commodity Channel Index (CCI)')
    ax2.set_ylabel('CCI')
    ax2.set_xlabel('Date')
    ax2.legend()
    plt.show()

def calculate_dmi(df, period=14):
    df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), df['high'] - df['high'].shift(1), 0)
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), df['low'].shift(1) - df['low'], 0)
    df['atr'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1).rolling(window=period).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()
    return df

def plot_dmi(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax2.plot(df['date'], df['plus_di'], label='+DI', color='green')
    ax2.plot(df['date'], df['minus_di'], label='-DI', color='red')
    ax2.plot(df['date'], df['adx'], label='ADX', color='blue')
    ax2.set_title('Directional Movement Index (DMI)')
    ax2.set_ylabel('DMI')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()
def calculate_vi(df, period=14):
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['tr_sum'] = df['tr'].rolling(window=period).sum()  
    df['vm_plus'] = abs(df['high'] - df['low'].shift(1))
    df['vm_minus'] = abs(df['low'] - df['high'].shift(1))
    df['vm_plus_sum'] = df['vm_plus'].rolling(window=period).sum()
    df['vm_minus_sum'] = df['vm_minus'].rolling(window=period).sum()
    df['vi_plus'] = df['vm_plus_sum'] / df['tr_sum']
    df['vi_minus'] = df['vm_minus_sum'] / df['tr_sum']
    return df

# Function to plot the Vortex Indicator
def plot_vi(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax2.plot(df['date'], df['vi_plus'], label='+VI', color='green')
    ax2.plot(df['date'], df['vi_minus'], label='-VI', color='red')
    ax2.set_title('Vortex Indicator (VI)')
    ax2.set_ylabel('VI')
    ax2.set_xlabel('Date')
    ax2.legend()
    plt.show()
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
    return ((value - mean) / std_dev)
#29
