import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib notebook

def plot_fibonacci_extension(df, price_column='Close', date_column='Date', start_date=None, end_date=None, retrace_date=None):
    if start_date is None or end_date is None or retrace_date is None:
        raise ValueError("Please provide start_date, end_date, and retrace_date.")

    start_price = df.loc[df[date_column] == start_date, price_column].values[0]
    end_price = df.loc[df[date_column] == end_date, price_column].values[0]
    retrace_price = df.loc[df[date_column] == retrace_date, price_column].values[0]

    trend_diff = end_price - start_price
    retrace_diff = retrace_price - start_price

    levels = {
        '0.0%': retrace_price,
        '61.8%': retrace_price + 0.618 * trend_diff,
        '100.0%': retrace_price + trend_diff,
        '161.8%': retrace_price + 1.618 * trend_diff,
        '200.0%': retrace_price + 2.0 * trend_diff,
        '261.8%': retrace_price + 2.618 * trend_diff
    }

    fig = go.Figure()

   
    fig.add_trace(go.Scatter(x=df[date_column], y=df[price_column], mode='lines', name='Close Price'))


    for level in levels:
        fig.add_trace(go.Scatter(x=[df[date_column].min(), df[date_column].max()], y=[levels[level], levels[level]],
                                 mode='lines', name=f'Fibonacci {level}', line={'dash': 'dash'}))

    fig.add_trace(go.Scatter(x=[start_date, end_date, retrace_date], y=[start_price, end_price, retrace_price],
                             mode='markers+text', name='Trend Points',
                             text=['Start', 'End', 'Retrace'],
                             textposition='top center'))

    fig.update_layout(
        title='Trend-Based Fibonacci Extension',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    fig.show()