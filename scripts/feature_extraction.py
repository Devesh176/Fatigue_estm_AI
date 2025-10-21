# 2_feature_extraction.py
import pandas as pd
import numpy as np

def compute_damage_indices(df):
    """
    Compute normalized micro (ρ) and macro (D) damage indices.
    ρ from PZT amplitude attenuation,
    D from stiffness loss (inverse strain ratio).
    """
    # --- Micro-scale: use baseline amplitude ---
    base_amp = df[df['cycle'] == df['cycle'].min()]['mean_amplitude'].iloc[0]
    df['rho_micro'] = 1 - (df['mean_amplitude'] / base_amp)
    df['rho_micro'] = df['rho_micro'].clip(lower=0)

    # --- Macro-scale: use median strain ratio ---
    base_strain = df[df['cycle'] == df['cycle'].min()]['median_strain'].iloc[0]
    df['D_macro'] = base_strain / df['median_strain']
    df['D_macro'] = df['D_macro'].clip(upper=1)

    # --- Smooth both curves ---
    df['rho_smooth'] = df['rho_micro'].ewm(span=3).mean()
    df['D_smooth']   = df['D_macro'].ewm(span=3).mean()

    return df[['cycle', 'rho_micro', 'rho_smooth', 'D_macro', 'D_smooth']]
