import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from error-handler import check_columns, log_function_call

@log_function_call
def alma(data, window_length, sigma=6.0, offset=0.85):
    m = offset * (window_length - 1)
    s = window_length / sigma
    q = np.arange(window_length)
    weights = np.exp(-q**2 / (2 * s**2))
    weights /= np.sum(weights)
    alma_values = np.convolve(data, weights[::-1], mode='valid')
    return alma_values

@log_function_call
def dema(data, window_length):
    ema1 = ema(data, window_length)
    ema2 = ema(ema1, window_length)
    dema_values = 2 * ema1 - ema2
    return dema_values

@log_function_call
def epma(data, window_length):
    alpha = 2 / (window_length + 1)
    epma_values = [np.nan] * (window_length - 1)
    for i in range(window_length - 1, len(data)):
        epma_values.append(alpha * data[i] + (1 - alpha) * epma_values[-1])
    return np.array(epma_values)

@log_function_call
def ema(data, window_length):
    alpha = 2 / (window_length + 1)
    ema_values = [np.nan] * (window_length - 1)
    for i in range(window_length - 1, len(data)):
        ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])
    return np.array(ema_values)

@log_function_call
def hilbert_transform_instantaneous_trendline(data, window_length):
    from scipy.signal import hilbert
    cycle = hilbert(np.array(data))
    trend = np.arctan(np.imag(cycle) / np.real(cycle))
    instantaneous_trendline = trend[(window_length - 1):]
    return instantaneous_trendline

@log_function_call
def hull_moving_average(data, window_length):
    wma = np.convolve(data, np.ones(window_length) / window_length, mode='valid')
    wma2 = np.convolve(wma, np.ones(int(np.sqrt(window_length))) / int(np.sqrt(window_length)), mode='valid')
    hma = wma2 * 2 - wma
    return hma

@log_function_call
def kaufmans_adaptive_moving_average(data, window_length, fast_length=2, slow_length=30):
    fast_ema = ema(data, fast_length)
    slow_ema = ema(data, slow_length)
    er = np.abs(fast_ema - slow_ema) / slow_ema
    sc = np.square(er).mean()
    kama = np.zeros_like(data)
    kama[window_length:] = [np.nan] * (window_length - 1) + [data[window_length - 1]]
    for i in range(window_length, len(data)):
        kama[i] = kama[i - 1] + sc * (data[i] - kama[i - 1])
    return kama[window_length:]

@log_function_call
def least_squares_moving_average(data, window_length):
    lsma = np.convolve(data, np.ones(window_length), mode='valid') / window_length
    weights = np.arange(1, window_length + 1)
    lsma = np.convolve(weights[::-1] * data, np.ones(window_length), mode='valid') / (np.arange(window_length, 0, -1).sum())
    return lsma

@log_function_call
def mesa_adaptive_moving_average(data, window_length, fast_limit=0.5, slow_limit=0.05):
    data = np.array(data)
    mama = np.zeros_like(data)
    mama[:window_length] = np.nan
    q = 0.5 / window_length
    n = 0.0962 * window_length ** 2 + 0.5769 * window_length - 0.2083
    n2 = 0.0962 * window_length ** 2 + 0.5769 * window_length - 0.3462
    j = 3 * window_length
    
    for i in range(window_length, len(data)):
        re = data[i] / data[i - window_length]
        re2 = re - 1
        re3 = abs(re2)
        
        if re3 < fast_limit:
            q = q + slow_limit
        else:
            q = q + re3
            
        mama[i] = mama[i - 1] + q * (data[i] - mama[i - 1])
        j = 0.0962 * j ** 2 + 0.5769 * j + n
        n2 = 0.0962 * n2 ** 2 + 0.5769 * n2 + n2
        q = q * (n / n2)
        
    return mama

@log_function_call
def mcginley_dynamic(data, window_length):
    data = np.array(data)
    mcginley = np.zeros_like(data)
    mcginley[:window_length] = np.nan
    c1 = 0.25 * (window_length + 1)
    c2 = 0.5 * window_length
    c3 = 0.25 * (window_length - 1)
    
    for i in range(window_length, len(data)):
        mcginley[i] = c1 * data[i] + c2 * mcginley[i - 1] + c3 * data[i - window_length + 1]
        
    return mcginley

@log_function_call
def modified_moving_average(data, window_length):
    data = np.array(data)
    mma = np.zeros_like(data)
    mma[:window_length] = np.nan
    sum_weights = 0
    
    for i in range(window_length):
        weight = (i + 1) * (window_length - i)
        mma[i + window_length - 1] = np.sum(data[i:window_length] * weight) / (window_length * (window_length + 1) / 2)
        sum_weights += weight

    for i in range(window_length, len(data)):
        mma[i] = mma[i - 1] + (data[i] - data[i - window_length]) / ((window_length + 1) / 2)
        
    return mma

@log_function_call
def running_moving_average(data, window_length):
    data = np.array(data)
    rma = np.zeros_like(data)
    rma[:window_length - 1] = np.nan
    rma[window_length - 1] = np.mean(data[:window_length])
    
    for i in range(window_length, len(data)):
        rma[i] = rma[i - 1] + (data[i] - data[i - window_length]) / window_length
        
    return rma

@log_function_call
def simple_moving_average(data, window_length):
    data = np.array(data)
    sma = np.convolve(data, np.ones(window_length) / window_length, mode='valid')
    return sma

@log_function_call
def smoothed_moving_average(data, window_length, smoothing_factor=2):
    data = np.array(data)
    smma = np.zeros_like(data)
    smma[:window_length] = np.nan
    smma[window_length - 1] = np.mean(data[:window_length])
    
    for i in range(window_length, len(data)):
        smma[i] = (smma[i - 1] * (window_length - 1) + data[i]) / window_length
        
    return smma

@log_function_call
def tillson_t3_moving_average(data, window_length):
    data = np.array(data)
    t3 = np.zeros_like(data)
    t3[:window_length] = np.nan
    ema1 = ema(data, window_length // 3)
    ema2 = ema(data, window_length // 2)
    ema3 = ema(data, window_length)
    
    for i in range(window_length, len(data)):
        t3[i] = ema3[i] + (ema3[i] - ema2[i]) + (ema2[i] - ema1[i])
        
    return t3

@log_function_call
def triple_exponential_moving_average(data, window_length):
    data = np.array(data)
    ema1 = ema(data, window_length)
    ema2 = ema(ema1, window_length)
    ema3 = ema(ema2, window_length)
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema

@log_function_call
@check_columns('Close')
def volume_weighted_average_price(data, volume):
    data = np.array(data)
    volume = np.array(volume)
    vwap = np.cumsum(data * volume) / np.cumsum(volume)
    return vwap

@log_function_call
@check_columns('Close')
def volume_weighted_moving_average(data, volume, window_length):
    data = np.array(data)
    volume = np.array(volume)
    vwma = np.convolve(data * volume, np.ones(window_length), mode='valid')/ np.convolve(volume, np.ones(window_length), mode='valid')
    return vwma

@log_function_call
def weighted_moving_average(data, window_length):
    data = np.array(data)
    weights = np.arange(1, window_length + 1)
    wma = np.convolve(data, weights, mode='valid') / weights.sum()
    return wma

@log_function_call
def FRAMA(close, length=14):
    half_length = int(length / 2)
    ema_half = ema(close, half_length)
    ema_full = ema(close, length)
    frama = ema_half + (0.5 * ema_half - ema_full) ** 4
    return frama

@log_function_call
def ZLEMA(close, length):
    half_length = int(length / 2)
    ema_half = ema(close, half_length)
    ema_double_half = ema(ema_half, half_length)
    zlema = (2 * ema_half - ema_double_half) ** 2
    return zlema

@log_function_call
@check_columns('close')
def tema(df, period=20):
    """Calculate Triple Exponential Moving Average (TEMA)."""
    ema1 = ema(df['close'], period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * (ema1 - ema2) + ema3
