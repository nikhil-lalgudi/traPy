import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}' with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' returned {result}")
        return result
    return wrapper

def exception_handling_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Exception occurred in function '{func.__name__}': {e}")
            return None
    return wrapper

%matplotlib inline 

class EndType(Enum):
    HIGH_LOW = 1
    CLOSE = 2

def get_pivots(df, left_span=2, right_span=2, max_trend_periods=20, end_type=EndType.HIGH_LOW):
    pivots = []
    for i in range(left_span, len(df) - right_span):
        if end_type == EndType.HIGH_LOW:
            is_pivot_high = True
            is_pivot_low = True
            for j in range(1, left_span + 1):
                if df['high'][i] <= df['high'][i - j] or df['high'][i] <= df['high'][i + j]:
                    is_pivot_high = False
                if df['low'][i] >= df['low'][i - j] or df['low'][i] >= df['low'][i + j]:
                    is_pivot_low = False
            if is_pivot_high:
                pivots.append((df['date'][i], df['high'][i], 'high'))
            if is_pivot_low:
                pivots.append((df['date'][i], df['low'][i], 'low'))
        elif end_type == EndType.CLOSE:
            is_pivot_close = True
            for j in range(1, left_span + 1):
                if df['close'][i] <= df['close'][i - j] or df['close'][i] <= df['close'][i + j]:
                    is_pivot_close = False
            if is_pivot_close:
                pivots.append((df['date'][i], df['close'][i], 'close'))
    return pivots

def plot_pivots(df, pivots):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.plot(df['date'], df['close'], label='Close Price', color='black')
    
    for pivot in pivots:
        if pivot[2] == 'high':
            ax.plot(pivot[0], pivot[1], marker='v', color='red', markersize=10)
        elif pivot[2] == 'low':
            ax.plot(pivot[0], pivot[1], marker='^', color='green', markersize=10)
        elif pivot[2] == 'close':
            ax.plot(pivot[0], pivot[1], marker='o', color='blue', markersize=10)
    
    ax.set_title('Pivot Points')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    plt.show()

def get_fractals(df, left_span=2, right_span=2, end_type=EndType.HIGH_LOW):
    fractal_bulls = [np.nan] * len(df)
    fractal_bears = [np.nan] * len(df)

    for i in range(left_span, len(df) - right_span):
        if end_type == EndType.HIGH_LOW:
            is_fractal_bull = all(df['high'][i] > df['high'][i - j] for j in range(1, left_span + 1)) and \
                              all(df['high'][i] > df['high'][i + j] for j in range(1, right_span + 1))
            is_fractal_bear = all(df['low'][i] < df['low'][i - j] for j in range(1, left_span + 1)) and \
                              all(df['low'][i] < df['low'][i + j] for j in range(1, right_span + 1))
            if is_fractal_bull:
                fractal_bulls[i] = df['high'][i]
            if is_fractal_bear:
                fractal_bears[i] = df['low'][i]
        elif end_type == EndType.CLOSE:
            is_fractal_bull = all(df['close'][i] > df['close'][i - j] for j in range(1, left_span + 1)) and \
                              all(df['close'][i] > df['close'][i + j] for j in range(1, right_span + 1))
            is_fractal_bear = all(df['close'][i] < df['close'][i - j] for j in range(1, left_span + 1)) and \
                              all(df['close'][i] < df['close'][i + j] for j in range(1, right_span + 1))
            if is_fractal_bull:
                fractal_bulls[i] = df['close'][i]
            if is_fractal_bear:
                fractal_bears[i] = df['close'][i]

    df['fractal_bull'] = fractal_bulls
    df['fractal_bear'] = fractal_bears
    return df

def plot_fractals(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(df['date'], df['close'], label='Close Price', color='black')
    ax.scatter(df['date'], df['fractal_bull'], label='Fractal Bull', color='green', marker='^', s=100, edgecolor='k')
    ax.scatter(df['date'], df['fractal_bear'], label='Fractal Bear', color='red', marker='v', s=100, edgecolor='k')
    
    ax.set_title('Fractals')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    plt.show()

def calculate_atr(df, period=14):
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df

def get_atr_stop(df, lookback_periods=21, multiplier=3, end_type=EndType.CLOSE):
    df = calculate_atr(df, period=lookback_periods)
    
    if end_type == EndType.CLOSE:
        df['atr_stop_long'] = df['close'] - (df['atr'] * multiplier)
        df['atr_stop_short'] = df['close'] + (df['atr'] * multiplier)
    elif end_type == EndType.HIGH_LOW:
        df['atr_stop_long'] = df['high'] - (df['atr'] * multiplier)
        df['atr_stop_short'] = df['low'] + (df['atr'] * multiplier)
    return df

def plot_atr_stop(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(df['date'], df['close'], label='Close Price', color='black')
    ax.plot(df['date'], df['atr_stop_long'], label='ATR Stop Long', color='green', linestyle='--')
    ax.plot(df['date'], df['atr_stop_short'], label='ATR Stop Short', color='red', linestyle='--')
    
    ax.set_title('ATR Stop')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    plt.show()

def calculate_aroon(df, lookback_period=25):
    df['aroon_up'] = df['high'].rolling(window=lookback_period + 1).apply(lambda x: (lookback_period - x[::-1].argmax()) / lookback_period * 100, raw=True)
    df['aroon_down'] = df['low'].rolling(window=lookback_period + 1).apply(lambda x: (lookback_period - x[::-1].argmin()) / lookback_period * 100, raw=True)
    return df

def plot_aroon(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df['date'], df['close'], label='Close Price', color='black')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax2.plot(df['date'], df['aroon_up'], label='Aroon Up', color='green')
    ax2.plot(df['date'], df['aroon_down'], label='Aroon Down', color='red')
    ax2.axhline(70, color='blue', linestyle='--', label='70 Threshold')
    ax2.axhline(30, color='blue', linestyle='--', label='30 Threshold')
    ax2.set_title('Aroon Indicator')
    ax2.set_ylabel('Aroon Value')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

# helper or just import from another file for far more efficient Abstraction
def calculate_tr(df):
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    return df

# helper
def calculate_dm(df):
    df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), df['high'] - df['high'].shift(1), 0)
    df['plus_dm'] = np.where(df['plus_dm'] < 0, 0, df['plus_dm'])
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), df['low'].shift(1) - df['low'], 0)
    df['minus_dm'] = np.where(df['minus_dm'] < 0, 0, df['minus_dm'])
    return df

def get_adx(df, lookback_periods=14):
    df = calculate_tr(df)
    df = calculate_dm(df)
    df['atr'] = df['tr'].rolling(window=lookback_periods).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=lookback_periods).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=lookback_periods).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=lookback_periods).mean()
    return df

def plot_adx(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(df['date'], df['close'], label='Close Price', color='black')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax2.plot(df['date'], df['adx'], label='ADX', color='blue')
    ax2.plot(df['date'], df['plus_di'], label='+DI', color='green')
    ax2.plot(df['date'], df['minus_di'], label='-DI', color='red')
    ax2.set_title('Average Directional Index (ADX)')
    ax2.set_ylabel('ADX Value')
    ax2.set_xlabel('Date')
    ax2.legend()
    plt.show()

#hurst is lowk chopped, check again
def hurst_exponent(time_series, max_lag=20, epsilon=1e-8):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    
    # Filter out zero or near-zero values
    valid_lags = [lags[i] for i in range(len(tau)) if tau[i] > epsilon]
    valid_tau = [tau[i] for i in range(len(tau)) if tau[i] > epsilon]
    
    if len(valid_lags) < 2 or len(valid_tau) < 2:
        raise ValueError("Not enough valid lag values to calculate Hurst Exponent.")
    
    poly = np.polyfit(np.log(valid_lags), np.log(valid_tau), 1)
    return poly[0] * 2.0

def plot_hurst_exponent(df, max_lag=20):
    try:
        hurst_value = hurst_exponent(df['close'], max_lag=max_lag)
    except ValueError as e:
        print(f"Error calculating Hurst Exponent: {e}")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['date'], df['close'], label='Close Price', color='black')
    ax.set_title(f'Hurst Exponent: {hurst_value:.2f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    plt.show()
    
    return hurst_value

## for later use abtraction

def calculate_atr(df, period=10):
    df['tr'] = np.maximum((df['high'] - df['low']), np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df

def calculate_supertrend(df, atr_period=10, multiplier=3):
    df = calculate_atr(df, period=atr_period)

    df['basic_upper_band'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
    df['basic_lower_band'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']
    

    df['supertrend_upper_band'] = df['basic_upper_band']
    df['supertrend_lower_band'] = df['basic_lower_band']
    
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > df['supertrend_upper_band'].iloc[i-1]:
            df.loc[i, 'supertrend_upper_band'] = max(df['basic_upper_band'].iloc[i], df['supertrend_upper_band'].iloc[i-1])
        else:
            df.loc[i, 'supertrend_upper_band'] = df['basic_upper_band'].iloc[i]
        
        if df['close'].iloc[i-1] < df['supertrend_lower_band'].iloc[i-1]:
            df.loc[i, 'supertrend_lower_band'] = min(df['basic_lower_band'].iloc[i], df['supertrend_lower_band'].iloc[i-1])
        else:
            df.loc[i, 'supertrend_lower_band'] = df['basic_lower_band'].iloc[i]
    
    df['supertrend'] = np.where(df['close'] > df['supertrend_upper_band'].shift(1), df['supertrend_upper_band'], df['supertrend_lower_band'])
    df['supertrend'] = np.where(df['close'] < df['supertrend_lower_band'].shift(1), df['supertrend_lower_band'], df['supertrend'])
    
    return df

def plot_supertrend(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['date'], df['close'], label='Close Price', color='black')
    ax.plot(df['date'], df['supertrend'], label='SuperTrend', color='green' if df['close'].iloc[-1] > df['supertrend'].iloc[-1] else 'red')
    
    ax.set_title('SuperTrend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    plt.show()

## for later use abtraction

def smoothed_moving_average(series, period, shift):
    sma = series.ewm(span=period, adjust=False).mean()
    return sma.shift(shift)

def calculate_alligator(df):
    df['jaw'] = smoothed_moving_average(df['close'], 13, 8)
    df['teeth'] = smoothed_moving_average(df['close'], 8, 5)
    df['lips'] = smoothed_moving_average(df['close'], 5, 3)
    return df


def plot_alligator(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['date'], df['close'], label='Close Price', color='black')
    ax.plot(df['date'], df['jaw'], label='Jaw (13, 8)', color='blue')
    ax.plot(df['date'], df['teeth'], label='Teeth (8, 5)', color='red')
    ax.plot(df['date'], df['lips'], label='Lips (5, 3)', color='green')
    
    ax.set_title('Williams Alligator')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    plt.show()

#9 