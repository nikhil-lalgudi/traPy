import pandas as pd
import numpy as np

def check_column_names(df):
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame. Ensure the DataFrame contains the columns: {required_columns}")
    return df