# scripts/create_labels.py
import pandas as pd

def add_rul_label(df):
    """
    Calculates the 'True RUL' label for a single coupon's dataframe.
    """
    if df.empty:
        return df
        
    # 1. Find the True End of Life (EOL)
    true_eol = df['cycle'].max()
    
    # 2. Calculate the RUL for every row
    df['RUL'] = true_eol - df['cycle']
    
    return df