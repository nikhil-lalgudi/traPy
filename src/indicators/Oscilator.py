import numpy as np

def awesome_oscillator(data, short_window=5, long_window=34):
    sma_short = np.convolve(data, np.ones(short_window) / short_window, mode='valid')
    sma_long = np.convolve(data, np.ones(long_window) / long_window, mode='valid')
    ao = sma_short - sma_long
    return ao

def chande_momentum_oscillator(data, window_length=20):
    data = np.array(data)
    gain = np.sum(np.maximum(0, data[window_length:] - data[:-window_length]))
    loss = np.sum(np.maximum(0, data[:-window_length] - data[window_length:]))
    cmo = 100 * (gain - loss) / (gain + loss)
    return cmo

def commodity_channel_index(data, window_length=20):
    data = np.array(data)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma = np.convolve(typical_price, np.ones(window_length) / window_length, mode='valid')
    mad = np.convolve(np.abs(typical_price - sma), np.ones(window_length), mode='valid') / window_length
    cci = (typical_price[window_length:] - sma) / (0.015 * mad)
    return cci

def connors_rsi(data, window_length=3, rsi_length=2):
    data = np.array(data)
    rsi = relative_strength_index(data, rsi_length)
    connors_rsi = np.zeros_like(rsi)
    connors_rsi[:window_length] = np.nan
    for i in range(window_length, len(rsi)):
        up = np.sum(np.maximum(0, rsi[i - window_length:i])) / window_length
        down = np.sum(np.minimum(0, rsi[i - window_length:i])) / window_length
        connors_rsi[i] = 100 * (1 + up / (1 - down))
    return connors_rsi

def detrended_price_oscillator(data, window_length=20):
    data = np.array(data)
    sma = np.convolve(data, np.ones(window_length) / window_length, mode='valid')
    dpo = data[window_length:] - sma
    return dpo

def kdj_index(data, k_window=9, d_window=3):
    data = np.array(data)
    high_low_range = data['High'].rolling(k_window).max() - data['Low'].rolling(k_window).min()
    k = 100 * (data['Close'] - data['Low'].rolling(k_window).min()) / (high_low_range + 1e-10)
    d = k.rolling(d_window).mean()
    j = 3 * d - 2 * k
    return k, d, j

def RSI(data, period = 14, column = 'Close'):
         delta = data[column].diff(1)
         delta = delta[1:]
         up = delta.copy()
         down = delta.copy()
         up[up <0] = 0
         down[down >0] = 0
         data['up'] = up
         data['down'] = down
         AVG_Gain = SMA(data, period, column = 'up')
         AVG_Loss = abs(SMA(data, period, column = 'down'))
         RS = AVG_Gain / AVG_Loss
         RSI = 100.0 - (100.0/(1.0+RS))
         
         data['RSI'] = RSI
         
         return data

def schaff_trend_cycle(data, window_length=23, cycle_length=10):
    data = np.array(data)
    ema1 = data['Close'].ewm(span=window_length, adjust=False).mean()
    ema2 = ema1.ewm(span=cycle_length, adjust=False).mean()
    schaff_trend_cycle = 100 * (ema1 - ema2) / ema2
    return schaff_trend_cycle

def stochastic_momentum_index(data, window_length=14, k_window=3, d_window=3):
    high_low_range = data['High'].rolling(window_length).max() - data['Low'].rolling(window_length).min()
    stochastic = (data['Close'] - data['Low'].rolling(window_length).min()) / (high_low_range + 1e-10)
    k = stochastic.rolling(k_window).mean()
    d = k.rolling(d_window).mean()
    smi = (k + d) / 2
    return k, d, smi

def stochastic_oscillator(data, k_window=14, d_window=3):
    high_low_range = data['High'].rolling(k_window).max() - data['Low'].rolling(k_window).min()
    k = 100 * (data['Close'] - data['Low'].rolling(k_window).min()) / (high_low_range + 1e-10)
    d = k.rolling(d_window).mean()
    return k, d

def stochastic_rsi(data, window_length=14, k_window=3, d_window=3):
    rsi = relative_strength_index(data, window_length)
    k = rsi.rolling(k_window).mean()
    d = k.rolling(d_window).mean()
    return k, d

def triple_ema_oscillator(data, window_length=9):
    ema1 = data['Close'].ewm(span=window_length, adjust=False).mean()
    ema2 = ema1.ewm(span=window_length, adjust=False).mean()
    ema3 = ema2.ewm(span=window_length, adjust=False).mean()
    trix = ema3.pct_change(periods=1)
    return trix

def ultimate_oscillator(data, window1=7, window2=14, window3=28):
    bp = data['Close'] - np.minimum(data['Low'].rolling(window1).min(), data['Low'].rolling(window1).min().shift(1))
    tr = np.maximum(data['High'], data['Close'].shift(1)) - np.minimum(data['Low'], data['Close'].shift(1))
    avg_bp = bp.rolling(window1).sum() / tr.rolling(window1).sum()
    avg_bp2 = avg_bp.rolling(window2).mean()
    avg_bp3 = avg_bp2.rolling(window3).mean()
    ultimate_oscillator = 100 * (4 * avg_bp3 + 2 * avg_bp2 + avg_bp) / (4 + 2 + 1)
    return ultimate_oscillator

def williams_r(data, window_length=14):
    high_low_range = data['High'].rolling(window_length).max() - data['Low'].rolling(window_length).min()
    williams_r = -100 * (data['High'] - data['Close']) / (high_low_range + 1e-10)
    return williams_r

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_eri(df, period=13):
    df['ema'] = calculate_ema(df['close'], period)
    df['bull_power'] = df['high'] - df['ema']
    df['bear_power'] = df['low'] - df['ema']
    return df

def plot_eri(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(df['date'], df['bull_power'], label='Bull Power', color='green')
    ax2.plot(df['date'], df['bear_power'], label='Bear Power', color='red')
    ax2.set_title('Elder-Ray Index')
    ax2.set_ylabel('Elder-Ray Index')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

def calculate_rvi(df, period=10):
    df['close_open'] = df['close'] - df['open']
    df['high_low'] = df['high'] - df['low']
    df['numerator'] = df['close_open'].rolling(window=period).mean()
    df['denominator'] = df['high_low'].rolling(window=period).mean()
    df['rvi'] = df['numerator'] / df['denominator']
    df['rvi_signal'] = df['rvi'].rolling(window=4).mean()
    return df

def plot_rvi(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax2.plot(df['date'], df['rvi'], label='RVI', color='blue')
    ax2.plot(df['date'], df['rvi_signal'], label='Signal Line', color='orange')
    ax2.set_title('Relative Vigor Index (RVI)')
    ax2.set_ylabel('RVI')
    ax2.set_xlabel('Date')
    ax2.legend()
    plt.show()

# SMAs but can be done with  abstraction
def smoothed_moving_average(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_gator_oscillator(df, jaw_period=13, teeth_period=8, lips_period=5):
    df['jaw'] = smoothed_moving_average(df['close'], jaw_period)
    df['teeth'] = smoothed_moving_average(df['close'], teeth_period)
    df['lips'] = smoothed_moving_average(df['close'], lips_period)
    df['upper_gator'] = df['jaw'] - df['teeth']
    df['lower_gator'] = df['teeth'] - df['lips']  
    return df

def plot_gator_oscillator(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df['date'], df['close'], label='Close Price', color='black')
    ax1.plot(df['date'], df['jaw'], label='Jaw (13-period SMA)', color='blue')
    ax1.plot(df['date'], df['teeth'], label='Teeth (8-period SMA)', color='red')
    ax1.plot(df['date'], df['lips'], label='Lips (5-period SMA)', color='green')
    ax1.set_title('Stock Price and Alligator Indicator')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.bar(df['date'], df['upper_gator'], label='Upper Gator', color='blue', alpha=0.7)
    ax2.bar(df['date'], -df['lower_gator'], label='Lower Gator', color='red', alpha=0.7)
    ax2.set_title('Gator Oscillator')
    ax2.set_ylabel('Gator Value')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

# 17