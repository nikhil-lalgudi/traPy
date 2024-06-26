import pandas as pd
import numpy as np

def check_columns(*required_columns):
    def decorator(func):
        def wrapper(data, *args, **kwargs):
            if isinstance(data, pd.DataFrame):
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"Missing columns in DataFrame: {missing_columns}")
            return func(data, *args, **kwargs)
        return wrapper
    return decorator

def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}' with arguments {args} and keyword arguments {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' completed")
        return result
    return wrapper

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

