import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

%matplotlib notebook

def plot_bollinger_bands(data, window=30):
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)

    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=window).std()

    data['%B'] = (data['Close'] - data['Lower Band']) / (data['Upper Band'] - data['Lower Band'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Middle Band'], label='Middle Band', color='black', linestyle='--')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')

    # Fill between bands
    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)

    # Candlestick-like bars
    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color='green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax2.plot(data.index, data['%B'], label='%B', color='blue')
    ax2.axhline(y=0, color='green', linestyle='--')
    ax2.axhline(y=1, color='red', linestyle='--')

    ax1.legend()
    ax1.set_title('Bollinger Bands')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('%B Indicator')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_donchian_channels(data, window=30):
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    data['Upper Band'] = data['High'].rolling(window=window).max()
    data['Lower Band'] = data['Low'].rolling(window=window).min()
    data['Middle Band'] = (data['Upper Band'] + data['Lower Band']) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Middle Band'], label='Middle Band', color='black', linestyle='--')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')

    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)

    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color 'green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax1.legend()
    ax1.set_title('Donchian Channels')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Volume')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_fractal_chaos_bands(data, window=30):
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    data['Upper Band'] = data['High'].rolling(window=window).max()
    data['Lower Band'] = data['Low'].rolling(window=window).min()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')

    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)

    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color='green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax1.legend()
    ax1.set_title('Fractal Chaos Bands')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Volume')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_keltner_channels(data, window=20, multiplier=2):
    # Ensure 'Date' is the index
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    # Calculate Keltner Channels
    data['Typical Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Middle Band'] = data['Typical Price'].rolling(window=window).mean()
    data['True Range'] = np.maximum((data['High'] - data['Low']),
                                    np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                               abs(data['Low'] - data['Close'].shift(1))))
    data['ATR'] = data['True Range'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] + (multiplier * data['ATR'])
    data['Lower Band'] = data['Middle Band'] - (multiplier * data['ATR'])


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Middle Band'], label='Middle Band', color='black', linestyle='--')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')

    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)

    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color='green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax1.legend()
    ax1.set_title('Keltner Channels')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Volume')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_moving_average_envelopes(data, window=20, percentage=2.5):
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] * (1 + (percentage / 100))
    data['Lower Band'] = data['Middle Band'] * (1 - (percentage / 100))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Middle Band'], label='Middle Band', color='black', linestyle='--')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')

    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)

    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color='green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax1.legend()
    ax1.set_title('Moving Average Envelopes')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Volume')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_starc_bands(data, window=15, multiplier=2):
    # Ensure 'Date' is the index
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    # Calculate STARC Bands
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['True Range'] = np.maximum((data['High'] - data['Low']),
                                    np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                               abs(data['Low'] - data['Close'].shift(1))))
    data['ATR'] = data['True Range'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] + (multiplier * data['ATR'])
    data['Lower Band'] = data['Middle Band'] - (multiplier * data['ATR'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Middle Band'], label='Middle Band', color='black', linestyle='--')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')

    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)

    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color='green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax1.legend()
    ax1.set_title('STARC Bands')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Volume')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_price_channels(data, window=20):
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    data['Upper Band'] = data['High'].rolling(window=window).max()
    data['Lower Band'] = data['Low'].rolling(window=window).min()
    data['Middle Band'] = (data['Upper Band'] + data['Lower Band']) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Middle Band'], label='Middle Band', color='black', linestyle='--')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')

    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)

    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color='green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax1.legend()
    ax1.set_title('Price Channels')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Volume')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Helper Function
def calculate_pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    resistance1 = (2 * pivot) - low
    support1 = (2 * pivot) - high
    resistance2 = pivot + (high - low)
    support2 = pivot - (high - low)
    resistance3 = high + 2 * (pivot - low)
    support3 = low - 2 * (high - pivot)
    
    return pd.DataFrame({
        'Pivot': pivot,
        'Resistance 1': resistance1,
        'Support 1': support1,
        'Resistance 2': resistance2,
        'Support 2': support2,
        'Resistance 3': resistance3,
        'Support 3': support3
    })

def plot_pivot_points(data):
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    pivots = calculate_pivot_points(data['High'], data['Low'], data['Close'])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data.index, data['Close'], label='Close', color='blue')
    ax.plot(pivots.index, pivots['Pivot'], label='Pivot', color='black', linestyle='--')
    ax.plot(pivots.index, pivots['Resistance 1'], label='Resistance 1', color='red', linestyle='--')
    ax.plot(pivots.index, pivots['Support 1'], label='Support 1', color='green', linestyle='--')
    ax.plot(pivots.index, pivots['Resistance 2'], label='Resistance 2', color='red', linestyle='--', alpha=0.7)
    ax.plot(pivots.index, pivots['Support 2'], label='Support 2', color='green', linestyle='--', alpha=0.7)
    ax.plot(pivots.index, pivots['Resistance 3'], label='Resistance 3', color='red', linestyle='--', alpha=0.5)
    ax.plot(pivots.index, pivots['Support 3'], label='Support 3', color='green', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title('Pivot Points')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True)
    plt.tight_layout()
    plt.show()

#Helper Function
def calculate_rolling_pivot_points(high, low, close, window=5):

    pivot = (high.rolling(window).mean() + low.rolling(window).mean() + close.rolling(window).mean()) / 3
    resistance1 = (2 * pivot) - low.rolling(window).mean()
    support1 = (2 * pivot) - high.rolling(window).mean()
    resistance2 = pivot + (high.rolling(window).mean() - low.rolling(window).mean())
    support2 = pivot - (high.rolling(window).mean() - low.rolling(window).mean())
    resistance3 = high.rolling(window).mean() + 2 * (pivot - low.rolling(window).mean())
    support3 = low.rolling(window).mean() - 2 * (high.rolling(window).mean() - pivot)
    
    return pd.DataFrame({
        'Pivot': pivot,
        'Resistance 1': resistance1,
        'Support 1': support1,
        'Resistance 2': resistance2,
        'Support 2': support2,
        'Resistance 3': resistance3,
        'Support 3': support3
    })

def plot_rolling_pivot_points(data, window=5):
  
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)

    pivots = calculate_rolling_pivot_points(data['High'], data['Low'], data['Close'], window)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(data.index, data['Close'], label='Close', color='blue')
    ax.plot(pivots.index, pivots['Pivot'], label='Pivot', color='black', linestyle='--')
    ax.plot(pivots.index, pivots['Resistance 1'], label='Resistance 1', color='red', linestyle='--')
    ax.plot(pivots.index, pivots['Support 1'], label='Support 1', color='green', linestyle='--')
    ax.plot(pivots.index, pivots['Resistance 2'], label='Resistance 2', color='red', linestyle='--', alpha=0.7)
    ax.plot(pivots.index, pivots['Support 2'], label='Support 2', color='green', linestyle='--', alpha=0.7)
    ax.plot(pivots.index, pivots['Resistance 3'], label='Resistance 3', color='red', linestyle='--', alpha=0.5)
    ax.plot(pivots.index, pivots['Support 3'], label='Support 3', color='green', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title('Rolling Pivot Points')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_standard_deviation_channels(data, window=20, num_std_dev=2):
 
    if 'Date' in data.columns:
        data.set_index('Date', inplace=True)
    
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Std Dev'] = data['Close'].rolling(window=window).std()
    data['Upper Band'] = data['Middle Band'] + (num_std_dev * data['Std Dev'])
    data['Lower Band'] = data['Middle Band'] - (num_std_dev * data['Std Dev'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.plot(data.index, data['Middle Band'], label='Middle Band', color='black', linestyle='--')
    ax1.plot(data.index, data['Upper Band'], label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, data['Lower Band'], label='Lower Band', color='green', linestyle='--')
    ax1.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='grey', alpha=0.3)
    for idx, row in data.iterrows():
        if row['Close'] >= data['Close'].shift(1).loc[idx]:
            ax1.plot([idx, idx], [data['Close'].shift(1).loc[idx], row['Close']], color='green')
        else:
            ax1.plot([idx, idx], [row['Close'], data['Close'].shift(1).loc[idx]], color='red')

    ax2.bar(data.index, data['Volume'], color='blue', alpha=0.3)

    ax1.legend()
    ax1.set_title('Standard Deviation Channels')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Volume')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
#10
