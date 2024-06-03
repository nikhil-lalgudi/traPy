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
def ichimoku_cloud(df, n1=9, n2=26, n3=52):
    # Calculate Tenkan-sen (Conversion Line)
    df['tenkan_sen'] = (df['high'].rolling(window=n1).max() + df['low'].rolling(window=n1).min()) / 2

    # Calculate Kijun-sen (Base Line)
    df['kijun_sen'] = (df['high'].rolling(window=n2).max() + df['low'].rolling(window=n2).min()) / 2

    # Calculate Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(n2)

    # Calculate Senkou Span B (Leading Span B)
    df['senkou_span_b'] = ((df['high'].rolling(window=n3).max() + df['low'].rolling(window=n3).min()) / 2).shift(n2)

    # Calculate Chikou Span (Lagging Span)
    df['chikou_span'] = df['close'].shift(-n2)
    return df

def plot_ichimoku_cloud(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close', color='black')

    # Plot Ichimoku Cloud components
    plt.plot(df.index, df['tenkan_sen'], label='Tenkan-sen (Conversion Line)', color='red')
    plt.plot(df.index, df['kijun_sen'], label='Kijun-sen (Base Line)', color='blue')
    plt.plot(df.index, df['senkou_span_a'], label='Senkou Span A (Leading Span A)', color='green')
    plt.plot(df.index, df['senkou_span_b'], label='Senkou Span B (Leading Span B)', color='orange')
    plt.plot(df.index, df['chikou_span'], label='Chikou Span (Lagging Span)', color='purple')

    # Fill the area between Senkou Span A and Senkou Span B to create the cloud
    plt.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] >= df['senkou_span_b'], facecolor='lightgreen', interpolate=True, alpha=0.5)
    plt.fill_between(df.index, df['senkou_span_a'], df['senkou_span_b'], where=df['senkou_span_a'] < df['senkou_span_b'], facecolor='lightcoral', interpolate=True, alpha=0.5)

    plt.title('Ichimoku Cloud')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
def calculate_parabolic_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
    high = df['high']
    low = df['low']
    close = df['close']

    # Initialize variables
    af = af_start
    ep = high[0]
    sar = low[0]
    uptrend = True
    psar = [sar]

    for i in range(1, len(df)):
        if uptrend:
            sar = sar + af * (ep - sar)
            sar = min(sar, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_increment, af_max)
            if low[i] < sar:
                uptrend = False
                sar = ep
                ep = low[i]
                af = af_start
        else:
            sar = sar + af * (ep - sar)
            sar = max(sar, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_increment, af_max)
            if high[i] > sar:
                uptrend = True
                sar = ep
                ep = high[i]
                af = af_start

        psar.append(sar)

    return pd.Series(psar, index=df.index)

def plot_parabolic_sar(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close', color='black')
    plt.plot(df.index, df['psar'], label='Parabolic SAR', linestyle='dashed', color='blue')
    plt.title('Parabolic SAR')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
#12
