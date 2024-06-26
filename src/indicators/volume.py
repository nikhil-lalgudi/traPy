from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from error-handler import log_execution_time

class DataPoint:
    def __init__(self, date, open, high, low, close, volume):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

@log_execution_time
def get_adl(DataPoints: List[DataPoint], sma_periods: Optional[int] = None):
    adl_values = []
    adl = 0
    for DataPoint in DataPoints:
        money_flow_multiplier = ((DataPoint.close - DataPoint.low) - (DataPoint.high - DataPoint.close)) / (DataPoint.high - DataPoint.low) if (DataPoint.high - DataPoint.low) != 0 else 0
        money_flow_volume = money_flow_multiplier * DataPoint.volume
        adl += money_flow_volume
        adl_values.append({
            'date': DataPoint.date,
            'adl': adl,
            'money_flow_multiplier': money_flow_multiplier,
            'money_flow_volume': money_flow_volume
        })

    if sma_periods is not None:
        for i in range(len(adl_values)):
            if i + 1 >= sma_periods:
                sum_adl = sum([adl_values[j]['adl'] for j in range(i + 1 - sma_periods, i + 1)])
                adl_values[i]['adl_sma'] = sum_adl / sma_periods
            else:
                adl_values[i]['adl_sma'] = None
    return adl_values

@log_execution_time
def get_cmf(DataPoints: List[DataPoint], lookback_periods: int = 20):
    cmf_values = []
    for i in range(len(DataPoints)):
        if i + 1 >= lookback_periods:
            sum_mfv = 0
            sum_volume = 0

            for j in range(i + 1 - lookback_periods, i + 1):
                DataPoint = DataPoints[j]
                money_flow_multiplier = ((DataPoint.close - DataPoint.low) - (DataPoint.high - DataPoint.close)) / (DataPoint.high - DataPoint.low) if (DataPoint.high - DataPoint.low) != 0 else 0
                money_flow_volume = money_flow_multiplier * DataPoint.volume
                sum_mfv += money_flow_volume
                sum_volume += DataPoint.volume

            cmf = sum_mfv / sum_volume if sum_volume != 0 else 0
        else:
            cmf = None
        cmf_values.append({
            'date': DataPoints[i].date,
            'cmf': cmf
        })
    return cmf_values

@log_execution_time
def calculate_ema(values: List[float], period: int) -> List[Optional[float]]:
    emas = []
    multiplier = 2 / (period + 1)
    ema = None
    
    for value in values:
        if ema is None:
            ema = value
        else:
            ema = (value - ema) * multiplier + ema
        emas.append(ema)
        
    return emas

@log_execution_time
def get_chaikin_osc(DataPoints: List[DataPoint], fast_periods: int = 3, slow_periods: int = 10):
    adl = []
    adl_sum = 0

    for DataPoint in DataPoints:
        money_flow_multiplier = ((DataPoint.close - DataPoint.low) - (DataPoint.high - DataPoint.close)) / (DataPoint.high - DataPoint.low) if (DataPoint.high - DataPoint.low) != 0 else 0
        money_flow_volume = money_flow_multiplier * DataPoint.volume
        adl_sum += money_flow_volume
        adl.append(adl_sum)
    fast_ema = calculate_ema(adl, fast_periods)
    slow_ema = calculate_ema(adl, slow_periods)
    chaikin_oscillator = [fast - slow if fast is not None and slow is not None else None for fast, slow in zip(fast_ema, slow_ema)]
    results = []
    for i, DataPoint in enumerate(DataPoints):
        results.append({
            'date': DataPoint.date,
            'oscillator': chaikin_oscillator[i],
            'adl': adl[i],
            'money_flow_multiplier': ((DataPoint.close - DataPoint.low) - (DataPoint.high - DataPoint.close)) / (DataPoint.high - DataPoint.low) if (DataPoint.high - DataPoint.low) != 0 else 0,
            'money_flow_volume': ((DataPoint.close - DataPoint.low) - (DataPoint.high - DataPoint.close)) / (DataPoint.high - DataPoint.low) * DataPoint.volume if (DataPoint.high - DataPoint.low) != 0 else 0
        })
    return results

@log_execution_time
def get_pvi(datapoints: List[DataPoint]):
    pvi_values = []
    pvi = 1000  # Starting value for PVI
    for i in range(1, len(datapoints)):
        if datapoints[i].volume > datapoints[i - 1].volume:
            pvi += (datapoints[i].close - datapoints[i - 1].close) / datapoints[i - 1].close * pvi
        pvi_values.append({
            'date': datapoints[i].date,
            'pvi': pvi
        })
    return pvi_values

@log_execution_time
def get_nvi(datapoints: List[DataPoint]):
    nvi_values = []
    nvi = 1000  # Starting value for NVI
    for i in range(1, len(datapoints)):
        if datapoints[i].volume < datapoints[i - 1].volume:
            nvi += (datapoints[i].close - datapoints[i - 1].close) / datapoints[i - 1].close * nvi
        nvi_values.append({
            'date': datapoints[i].date,
            'nvi': nvi
        })
    return nvi_values

@log_execution_time
def calculate_eom(df, period=14):
    df['midpoint_move'] = ((df['high'] + df['low']) / 2).diff()
    df['box_ratio'] = (df['volume'] / 10000) / (df['high'] - df['low'])
    df['eom'] = df['midpoint_move'] / df['box_ratio']
    df['eom_sma'] = df['eom'].rolling(window=period).mean()
    return df

@log_execution_time
def plot_eom(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    # Plot OHLC data
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot EOM indicator
    ax2.plot(df['date'], df['eom_sma'], label='EOM (14)', color='brown')
    ax2.axhline(0, color='blue', linewidth=0.5)
    ax2.fill_between(df['date'], df['eom_sma'], 0, where=(df['eom_sma'] > 0), facecolor='green', alpha=0.3, interpolate=True, label='Advancing with Ease')
    ax2.fill_between(df['date'], df['eom_sma'], 0, where=(df['eom_sma'] < 0), facecolor='red', alpha=0.3, interpolate=True, label='Declining with Ease')

    ax2.set_title('Ease of Movement (EOM)')
    ax2.set_ylabel('EOM')
    ax2.set_xlabel('Date')
    ax2.legend()

    # Highlight regions
    advancing_region = df[(df['eom_sma'] > 0)]
    declining_region = df[(df['eom_sma'] < 0)]

    for region in [advancing_region, declining_region]:
        if not region.empty:
            start = region.index[0]
            end = region.index[-1]
            ax2.axvspan(df['date'][start], df['date'][end], color='yellow', alpha=0.1)
    plt.show()

@log_execution_time
def calculate_adl(df):
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_volume = mf_multiplier * df['volume']
    df['adl'] = mf_volume.cumsum()
    return df

@log_execution_time
def plot_acc_dist(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot OHLC data
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot ADL indicator
    ax2.plot(df['date'], df['adl'], label='Accum/Dist', color='olive')
    ax2.set_title('Accumulation/Distribution Line')
    ax2.set_ylabel('ADL')
    ax2.set_xlabel('Date')
    ax2.legend()

    # Highlight divergences (for example purposes, highlight a range)
    ax1.annotate('Bearish Divergence', xy=(df['date'][30], df['close'][30]), 
                 xytext=(df['date'][20], df['close'][20] + 10),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red')

    ax2.annotate('Bearish Divergence', xy=(df['date'][30], df['adl'][30]), 
                 xytext=(df['date'][20], df['adl'][20] + 10),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red')

    # Example lines indicating divergence (manual example, adapt to actual data)
    ax1.plot([df['date'][20], df['date'][30]], [df['close'][20], df['close'][30]], color='blue')
    ax2.plot([df['date'][20], df['date'][30]], [df['adl'][20], df['adl'][30]], color='blue')

    plt.show()

@log_execution_time
def calculate_vpt(df):
    vpt = [0]  # VPT starts at 0
    for i in range(1, len(df)):
        vpt.append(vpt[-1] + (df['close'][i] - df['close'][i-1]) / df['close'][i-1] * df['volume'][i])
    df['vpt'] = vpt
    return df

@log_execution_time
def plot_vpt(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    # Plot OHLC data
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    # Plot VPT indicator
    ax2.plot(df['date'], df['vpt'], label='VPT', color='blue')
    ax2.set_title('Volume Price Trend (VPT)')
    ax2.set_ylabel('VPT')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

@log_execution_time
def calculate_obv(df):
    obv = [0]  # OBV starts at 0
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i - 1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close'][i] < df['close'][i - 1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    return df

@log_execution_time
def plot_obv(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    # Plot OHLC data
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    # Plot OBV indicator
    ax2.plot(df['date'], df['obv'], label='OBV', color='purple')
    ax2.set_title('On-Balance Volume (OBV)')
    ax2.set_ylabel('OBV')
    ax2.set_xlabel('Date')
    ax2.legend()

    ax1.annotate('Prices going Up', xy=(df['date'][40], df['close'][40]), 
                 xytext=(df['date'][30], df['close'][30] + 10),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=12, color='green')

    ax2.annotate('OBV Rising', xy=(df['date'][40], df['obv'][40]), 
                 xytext=(df['date'][30], df['obv'][30] - 5000),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=12, color='green')
# FIX THE ARROWS 
    
    ax1.plot([df['date'][30], df['date'][40]], [df['close'][30], df['close'][40]], color='blue')
    ax2.plot([df['date'][30], df['date'][40]], [df['obv'][30], df['obv'][40]], color='blue')

    plt.show()

@log_execution_time
def calculate_pvo(df, fast_periods=12, slow_periods=26, signal_periods=9):
    df['ema_fast'] = calculate_ema(df['volume'], fast_periods)
    df['ema_slow'] = calculate_ema(df['volume'], slow_periods)
    df['pvo'] = ((df['ema_fast'] - df['ema_slow']) / df['ema_slow']) * 100
    df['signal'] = calculate_ema(df['pvo'], signal_periods)
    df['histogram'] = df['pvo'] - df['signal']
    return df

@log_execution_time
def plot_pvo(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(df['date'], df['pvo'], label='PVO', color='blue')
    ax2.plot(df['date'], df['signal'], label='Signal', color='orange')
    ax2.bar(df['date'], df['histogram'], label='Histogram', color='gray')
    ax2.set_title('Percentage Volume Oscillator (PVO)')
    ax2.set_ylabel('PVO')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

@log_execution_time
def calculate_force_index(df, lookback_periods):
    df['force_index'] = df['close'].diff() * df['volume']
    df['force_index_ema'] = df['force_index'].ewm(span=lookback_periods, adjust=False).mean()
    return df

@log_execution_time
def plot_force_index(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(df['date'], df['force_index_ema'], label='Force Index', color='blue')
    ax2.set_title('Force Index')
    ax2.set_ylabel('Force Index')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

@log_execution_time
def calculate_mfi(df, lookback_periods=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = [0] * len(df)
    negative_flow = [0] * len(df)

    for i in range(1, len(df)):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            negative_flow[i] = money_flow[i]

    positive_flow = pd.Series(positive_flow).rolling(window=lookback_periods).sum()
    negative_flow = pd.Series(negative_flow).rolling(window=lookback_periods).sum()
    
    mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
    df['mfi'] = mfi
    return df

@log_execution_time
def plot_mfi(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    
    ax2.plot(df['date'], df['mfi'], label='MFI', color='green')
    ax2.axhline(20, color='red', linestyle='--')
    ax2.axhline(80, color='red', linestyle='--')
    ax2.set_title('Money Flow Index (MFI)')
    ax2.set_ylabel('MFI')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

@log_execution_time
def calculate_kvo(df, fast_periods=34, slow_periods=55, signal_periods=13):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume_force'] = df['volume'] * df['typical_price'].diff().fillna(0)
    df['fast_kvo'] = calculate_ema(df['volume_force'], fast_periods)
    df['slow_kvo'] = calculate_ema(df['volume_force'], slow_periods)
    df['oscillator'] = df['fast_kvo'] - df['slow_kvo']
    df['signal'] = calculate_ema(df['oscillator'], signal_periods)
    return df

@log_execution_time
def plot_kvo(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax2.plot(df['date'], df['oscillator'], label='KVO Oscillator', color='blue')
    ax2.plot(df['date'], df['signal'], label='Signal Line', color='orange')
    ax2.set_title('Klinger Volume Oscillator (KVO)')
    ax2.set_ylabel('KVO')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.show()

@log_execution_time
def calculate_vwap(df):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cum_volume'] = df['volume'].cumsum()
    df['cum_volume_price'] = (df['typical_price'] * df['volume']).cumsum()
    df['vwap'] = df['cum_volume_price'] / df['cum_volume']
    return df

@log_execution_time
def plot_vwap(df):
    fig, ax1 = plt.subplots(figsize=(14, 10))
    # Plot OHLC data
    ax1.plot(df['date'], df['close'], label='Close Price')
    ax1.plot(df['date'], df['vwap'], label='VWAP', color='orange')
    ax1.set_title('Stock Price with VWAP')
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    ax1.legend()
    plt.show()

@log_execution_time
def calculate_twiggs_money_flow(df, period):
    df['true_high'] = df[['high', 'close']].max(axis=1)
    df['true_low'] = df[['low', 'close']].min(axis=1)
    df['mfm'] = ((df['close'] - df['true_low']) - (df['true_high'] - df['close'])) / (df['true_high'] - df['true_low'])
    # Handle division by zero (when true_high == true_low)
    df['mfm'] = df['mfm'].replace([float('inf'), -float('inf')], 0).fillna(0) 
    df['mfv'] = df['mfm'] * df['volume'] 
    df['tmf'] = df['mfv'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return df['tmf']

@log_execution_time
def plot_twiggs_money_flow(df, period=3):
    df['tmf'] = calculate_twiggs_money_flow(df, period)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['tmf'], label='Twiggs Money Flow', color='blue')
    plt.title('Twiggs Money Flow (TMF)')
    plt.xlabel('Date')
    plt.ylabel('TMF')
    plt.legend()
    plt.grid(True)
    plt.show()