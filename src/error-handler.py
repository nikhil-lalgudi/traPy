import pandas as pd
import numpy as np

def check_column_names(df):
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame. Ensure the DataFrame contains the columns: {required_columns}")
    return df

def check_numeric_or_single_arg_callable(value, name):
    if not (isinstance(value, (int, float)) or callable(value)):
        raise ValueError(f"{name} must be a numeric value or a callable accepting a single argument")

def ensure_single_arg_constant_function(value):
    if callable(value):
        return value
    return lambda _: value

def check_positive_integer(value):
    if not (isinstance(value, int) and value > 0):
        raise ValueError("Value must be a positive integer")

def check_numeric(value, name):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

