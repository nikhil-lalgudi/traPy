import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

def plot_fibonacci_retracement(df, price_column='Close', date_column='Date'):
    min_price = df[price_column].min()
    max_price = df[price_column].max()

    diff = max_price - min_price
    levels = {
        '0.0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50.0%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100.0%': min_price
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df[date_column], y=df[price_column], mode='lines', name='Close Price'))

    for level in levels:
        fig.add_trace(go.Scatter(x=[df[date_column].min(), df[date_column].max()], y=[levels[level], levels[level]],
                                 mode='lines', name=f'Fibonacci {level}', line={'dash': 'dash'}))

    fig.update_layout(
        title='Fibonacci Retracement',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )

    fig.show()

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

def plot_fibonacci_channel(df, price_column='Close', date_column='Date', start_date=None, end_date=None):

    if start_date is None or end_date is None:
        raise ValueError("Please provide start_date and end_date.")

    start_price = df.loc[df[date_column] == start_date, price_column].values[0]
    end_price = df.loc[df[date_column] == end_date, price_column].values[0]

    trend_diff = end_price - start_price

    levels = {
        '0.0%': start_price,
        '23.6%': start_price + 0.236 * trend_diff,
        '38.2%': start_price + 0.382 * trend_diff,
        '50.0%': start_price + 0.5 * trend_diff,
        '61.8%': start_price + 0.618 * trend_diff,
        '100.0%': end_price
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df[date_column], y=df[price_column], mode='lines', name='Close Price'))

    for level in levels:
        fig.add_trace(go.Scatter(x=[start_date, end_date], y=[start_price, levels[level]],
                                 mode='lines', name=f'Fibonacci {level}', line={'dash': 'dash'}))
        fig.add_trace(go.Scatter(x=[df[date_column].min(), df[date_column].max()], y=[levels[level], levels[level]],
                                 mode='lines', name=f'Fibonacci {level}', line={'dash': 'dash'}))

    fig.add_trace(go.Scatter(x=[start_date, end_date], y=[start_price, end_price],
                             mode='markers+text', name='Trend Points',
                             text=['Start', 'End'],
                             textposition='top center'))

    fig.update_layout(
        title='Fibonacci Channel',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )

    fig.show()

def run_fibonacci_channel_app():
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='fib-channel-graph'),
        html.Div(id='selected-points', style={'display': 'none'}),
        html.Div([
            html.Button('Draw Fibonacci Channel', id='draw-button', n_clicks=0)
        ]),
        dcc.Graph(id='output-graph')
    ])

    @app.callback(
        Output('fib-channel-graph', 'figure'),
        [Input('selected-points', 'children')]
    )
    def update_graph(selected_points):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))

        fig.update_layout(
            title='Select Start and End Points',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark'
        )
        return fig

    @app.callback(
        Output('selected-points', 'children'),
        [Input('fib-channel-graph', 'clickData')],
        [State('selected-points', 'children')]
    )
    def select_points(clickData, selected_points):
        if clickData:
            if selected_points is None:
                selected_points = []
            selected_points.append(clickData['points'][0]['x'])
            if len(selected_points) > 2:
                selected_points = selected_points[-2:]
        return selected_points

    @app.callback(
        Output('output-graph', 'figure'),
        [Input('draw-button', 'n_clicks')],
        [State('selected-points', 'children')]
    )
    def draw_fibonacci_channel(n_clicks, selected_points):
        if n_clicks > 0 and selected_points and len(selected_points) == 2:
            start_date = selected_points[0]
            end_date = selected_points[1]

            start_price = df.loc[df['Date'] == start_date, 'Close'].values[0]
            end_price = df.loc[df['Date'] == end_date, 'Close'].values[0]
            trend_diff = end_price - start_price
            levels = {
                '0.0%': start_price,
                '23.6%': start_price + 0.236 * trend_diff,
                '38.2%': start_price + 0.382 * trend_diff,
                '50.0%': start_price + 0.5 * trend_diff,
                '61.8%': start_price + 0.618 * trend_diff,
                '100.0%': end_price
            }
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
            for level in levels:
                fig.add_trace(go.Scatter(x=[start_date, end_date], y=[start_price, levels[level]],
                                         mode='lines', name=f'Fibonacci {level}', line={'dash': 'dash'}))
                fig.add_trace(go.Scatter(x=[df['Date'].min(), df['Date'].max()], y=[levels[level], levels[level]],
                                         mode='lines', name=f'Fibonacci {level}', line={'dash': 'dash'}))
            fig.add_trace(go.Scatter(x=[start_date, end_date], y=[start_price, end_price],
                                     mode='markers+text', name='Trend Points',
                                     text=['Start', 'End'],
                                     textposition='top center'))
            fig.update_layout(
                title='Fibonacci Channel',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark'
            )

            return fig
        return go.Figure()

    app.run_server(debug=True)

if __name__ == '__main__':
    run_fibonacci_channel_app()
