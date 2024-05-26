import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ehlers_fisher_transform(df, period=10):
    # Calculate the median price
    df['Median Price'] = (df['High'] + df['Low']) / 2

    df['Fisher Transform'] = 0.0
    df['Trigger'] = 0.0

    df['Max High'] = df['High'].rolling(window=period).max()
    df['Min Low'] = df['Low'].rolling(window=period).min()
    df['Normalized Price'] = 2 * ((df['Median Price'] - df['Min Low']) / (df['Max High'] - df['Min Low']) - 0.5)
    
    # Apply Fisher Transform
    df['Fisher Transform'] = 0.5 * np.log((1 + df['Normalized Price']) / (1 - df['Normalized Price']))
    df['Fisher Transform'] = df['Fisher Transform'].ewm(span=period, adjust=False).mean()
    
    # Create the trigger line
    df['Trigger'] = df['Fisher Transform'].shift(1)
    return df[['Fisher Transform', 'Trigger']]