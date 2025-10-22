# scripts/2_feature_extraction.py
import pandas as pd
import numpy as np

def compute_damage_indices(df):
    """
    Compute normalized micro (ρ) and macro (D) damage indices.
    ρ from PZT amplitude attenuation, 
    D from stiffness loss (inverse strain ratio). 
    """
    
    if df.empty:
        return df
        
    # --- Micro-scale: use baseline amplitude --- 
    # Get the first available amplitude as baseline
    base_amp = df['mean_amplitude'].iloc[0] 

    print("base_amp", base_amp)

    df['rho_micro'] = 1 - (df['mean_amplitude'] / base_amp) 
    df['rho_micro'] = df['rho_micro'].clip(lower=0.0, upper=1.0) 

    # --- Macro-scale: use median strain ratio --- 
    # Get the first available strain as baseline
    base_strain = df['median_strain'].iloc[0] 
    df['D_macro'] = base_strain / df['median_strain'] 
    df['D_macro'] = df['D_macro'].clip(lower=0.0, upper=1.0) 

    # --- Smooth both curves --- 
    df['rho_smooth'] = df['rho_micro'].ewm(span=3).mean() 
    df['D_smooth']   = df['D_macro'].ewm(span=3).mean() 

    return df