import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Decorator for logging function calls
def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}' with arguments {args} and keyword arguments {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' completed")
        return result
    return wrapper

# Decorator for checking required columns
def check_columns(*required_columns):
    def decorator(func):
        def wrapper(df, *args, **kwargs):
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in DataFrame: {missing_columns}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

@log_function_call
@check_columns('High', 'Low', 'Close')
def ehlers_fisher_transform(df, period=10):
    df['Median Price'] = (df['High'] + df['Low']) / 2

    df['Fisher Transform'] = 0.0
    df['Trigger'] = 0.0

    df['Max High'] = df['High'].rolling(window=period).max()
    df['Min Low'] = df['Low'].rolling(window=period).min()
    df['Normalized Price'] = 2 * ((df['Median Price'] - df['Min Low']) / (df['Max High'] - df['Min Low']) - 0.5)
    
    df['Fisher Transform'] = 0.5 * np.log((1 + df['Normalized Price']) / (1 - df['Normalized Price']))
    df['Fisher Transform'] = df['Fisher Transform'].ewm(span=period, adjust=False).mean()
    
    df['Trigger'] = df['Fisher Transform'].shift(1)
    return df[['Fisher Transform', 'Trigger']]

@log_function_call
@check_columns('High', 'Low', 'Close')
def ehlers_fisher_transform_plot(df, period=10):
    df = ehlers_fisher_transform(df, period)
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10), sharex=True)
    ax1.plot(df.index, df['Median Price'], label='Median Price')
    ax1.set_title('Price Data')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
 
    ax2.plot(df.index, df['Fisher Transform'], label='Fisher Transform', color='blue')
    ax2.plot(df.index, df['Trigger'], label='Trigger', color='red')
    ax2.set_title('Ehlers Fisher Transform')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

@log_function_call
@check_columns('Open', 'High', 'Low', 'Close')
def heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_df['HA_Open'] = 0.0
    ha_df['HA_Open'].iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(ha_df)):
        ha_df['HA_Open'].iloc[i] = (ha_df['HA_Open'].iloc[i - 1] + ha_df['HA_Close'].iloc[i - 1]) / 2
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
    return ha_df

@log_function_call
@check_columns('Open', 'High', 'Low', 'Close')
def heikin_ashi_plot(df):  
    ha_df = heikin_ashi(df)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10), sharex=True)
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.set_title('Original Price Data')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(ha_df.index, ha_df['HA_Close'], label='Heikin-Ashi Close', color='blue')
    ax2.plot(ha_df.index, ha_df['HA_Open'], label='Heikin-Ashi Open', color='green')
    ax2.plot(ha_df.index, ha_df['HA_High'], label='Heikin-Ashi High', color='red')
    ax2.plot(ha_df.index, ha_df['HA_Low'], label='Heikin-Ashi Low', color='orange')
    ax2.set_title('Heikin-Ashi')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

@log_function_call
@check_columns('Close')
def renko(df, brick_size):
    renko_df = pd.DataFrame(columns=['Date', 'Price', 'Brick'])
    renko_df['Date'] = df.index
    renko_df['Price'] = df['Close']
    
    renko_bricks = []
    brick = 0
    direction = 0  # 1 for up, -1 for down
    for price in df['Close']:
        if direction == 0:
            brick = price
            renko_bricks.append(brick)
            direction = 1 if price >= brick + brick_size else -1
        elif direction == 1:
            if price >= brick + brick_size:
                brick += brick_size
                renko_bricks.append(brick)
                direction = 1
            elif price <= brick - brick_size:
                brick -= brick_size
                renko_bricks.append(brick)
                direction = -1
        elif direction == -1:
            if price <= brick - brick_size:
                brick -= brick_size
                renko_bricks.append(brick)
                direction = -1
            elif price >= brick + brick_size:
                brick += brick_size
                renko_bricks.append(brick)
                direction = 1
                
    renko_df['Brick'] = renko_bricks[:len(renko_df)]
    return renko_df

@log_function_call
@check_columns('Close')
def renko_chart(df, brick_size):
    renko_df = pd.DataFrame(columns=['Date', 'Close'])
    renko_df['Date'] = df.index
    renko_df['Close'] = df['Close']
    renko_df.set_index('Date', inplace=True)
    
    renko_prices = []
    renko_dates = []
    uptrend = True
    
    previous_close = df['Close'].iloc[0]
    for date, close in df['Close'].items():
        move = close - previous_close
        
        if uptrend:
            if move >= brick_size:
                bricks = int(move // brick_size)
                renko_prices.extend([previous_close + brick_size * i for i in range(1, bricks + 1)])
                renko_dates.extend([date] * bricks)
                previous_close += bricks * brick_size
            elif move <= -brick_size:
                bricks = int(-move // brick_size)
                renko_prices.extend([previous_close - brick_size * i for i in range(1, bricks + 1)])
                renko_dates.extend([date] * bricks)
                previous_close -= bricks * brick_size
                uptrend = False
        else:
            if move <= -brick_size:
                bricks = int(-move // brick_size)
                renko_prices.extend([previous_close - brick_size * i for i in range(1, bricks + 1)])
                renko_dates.extend([date] * bricks)
                previous_close -= bricks * brick_size
            elif move >= brick_size:
                bricks = int(move // brick_size)
                renko_prices.extend([previous_close + brick_size * i for i in range(1, bricks + 1)])
                renko_dates.extend([date] * bricks)
                previous_close += bricks * brick_size
                uptrend = True

    renko_df = pd.DataFrame({'Date': renko_dates, 'Close': renko_prices}).set_index('Date')
    
    return renko_df

@log_function_call
@check_columns('Close')
def renko_chart_plot(df, brick_size):
    renko_df = renko_chart(df, brick_size)
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10), sharex=True)
    
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.set_title('Original Price Data')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(renko_df.index, renko_df['Close'], label='Renko Close', color='blue')
    ax2.set_title('Renko Chart')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

@log_function_call
@check_columns('Close')
def zigzag_indicator(df, pct_change):
    zigzag_df = pd.DataFrame(columns=['Date', 'Close', 'ZigZag'])
    zigzag_df['Date'] = df.index
    zigzag_df['Close'] = df['Close']
    zigzag_df.set_index('Date', inplace=True)
    
    last_pivot = df['Close'][0]
    last_pivot_idx = df.index[0]
    uptrend = None
    zigzag_values = []

    for date, close in df['Close'].items():
        move_pct = (close - last_pivot) / last_pivot
        
        if uptrend is None:
            uptrend = move_pct > 0
            zigzag_values.append(last_pivot)
        elif uptrend and move_pct <= -pct_change:
            zigzag_values.append(last_pivot)
            last_pivot = close
            last_pivot_idx = date
            uptrend = False
        elif not uptrend and move_pct >= pct_change:
            zigzag_values.append(last_pivot)
            last_pivot = close
            last_pivot_idx = date
            uptrend = True
        else:
            zigzag_values.append(None)
    
    zigzag_values[-1] = df['Close'].iloc[-1]  # Ensure the last value is included
    zigzag_df['ZigZag'] = zigzag_values
    return zigzag_df

@log_function_call
@check_columns('Close')
def zigzag_indicator_plot(df, pct_change):
    zigzag_df = zigzag_indicator(df, pct_change) 
    fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 10), sharex=True)
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.set_title('Original Price Data')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
   
    ax2.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    ax2.plot(zigzag_df.index, zigzag_df['ZigZag'], label='Zig Zag', color='red')
    ax2.set_title('Zig Zag Indicator')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)  
    plt.show()

@log_function_call
@check_columns('close')
def calculate_kagi(df, reversal_amount=0.04):
    kagi = []
    direction = None
    for i in range(1, len(df)):
        if direction is None:
            kagi.append(df['close'][i])
            direction = 'up' if df['close'][i] > df['close'][i - 1] else 'down'
        else:
            if direction == 'up':
                if df['close'][i] >= kagi[-1]:
                    kagi.append(df['close'][i])
                elif df['close'][i] <= kagi[-1] * (1 - reversal_amount):
                    kagi.append(df['close'][i])
                    direction = 'down'
            else:
                if df['close'][i] <= kagi[-1]:
                    kagi.append(df['close'][i])
                elif df['close'][i] >= kagi[-1] * (1 + reversal_amount):
                    kagi.append(df['close'][i])
                    direction = 'up'
    df['kagi'] = kagi + [np.nan] * (len(df) - len(kagi))
    return df

@log_function_call
@check_columns('date', 'kagi')
def plot_kagi(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(df['date'], df['kagi'], drawstyle='steps-post', label='Kagi Chart')
    ax.set_title('Kagi Chart')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    plt.show()

@log_function_call
@check_columns('close')
def calculate_line_break(df, num_lines=3):
    lines = []
    direction = None
    for i in range(1, len(df)):
        if direction is None:
            lines.append(df['close'][i])
            direction = 'up' if df['close'][i] > df['close'][i - 1] else 'down'
        else:
            if direction == 'up':
                if df['close'][i] > max(lines[-num_lines:]):
                    lines.append(df['close'][i])
                    direction = 'up'
                elif df['close'][i] < min(lines[-num_lines:]):
                    lines.append(df['close'][i])
                    direction = 'down'
            else:
                if df['close'][i] < min(lines[-num_lines:]):
                    lines.append(df['close'][i])
                    direction = 'down'
                elif df['close'][i] > max(lines[-num_lines:]):
                    lines.append(df['close'][i])
                    direction = 'up'
    df['line_break'] = lines + [np.nan] * (len(df) - len(lines))
    return df

@log_function_call
@check_columns('date', 'line_break')
def plot_line_break(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(df['date'], df['line_break'], drawstyle='steps-post', label='Line Break Chart')
    ax.set_title('Line Break Chart')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    plt.show()

#6
## error handling must be done for 'close' to 'Close', 'high' to 'High', 'low' to 'Low', 'open' to 'Open' etc