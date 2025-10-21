# main.py (Corrected and Updated)
import pandas as pd
import numpy as np
from scripts.load_data import aggregate_coupon_data
from scripts.feature_extraction import compute_damage_indices
from scripts.bayesian_filter import ParticleFilter
from scripts.utils import physics_state_transition # This is the model
from scripts.rul_prediction import predict_RUL_series # This runs the whole process
from scripts.visualize_results import plot_all_results # This plots everything

# --- 1. Load and process data ---
print("Loading and processing data...")
df = aggregate_coupon_data("data/PZT-data", "data/StrainData")
df = compute_damage_indices(df)

# Drop rows with NaN in the smoothed data (usually first few)
df = df.dropna(subset=['rho_smooth', 'D_smooth'])

print("Data loaded. Head:")
print(df[['cycle', 'rho_smooth', 'D_smooth']].head())


# --- 2. Define Particle Filter Configuration ---
# Initial guesses for parameters (A_t, alpha_t) from Fig 2d
init_params_mean = [8.5e-4, 1.8] 
init_params_cov = np.diag([1e-7, 0.1]) # Initial uncertainty

# Initial state (rho=0, D=1)
init_state = [0.0, 1.0] 

# Full PF configuration dictionary
pf_config = {
    'N': 1000, # Number of particles (more is better but slower)
    'sigma_state_rho': 1e-3,
    'sigma_state_D': 1e-4,
    'sigma_param_At': 1e-7,
    'sigma_param_alpha_t': 1e-4,
    'init_state': init_state,
    'init_params_mean': init_params_mean,
    'init_params_cov': init_params_cov
}


# --- 3. Run Filter & Prognosis ---
# This new function runs the PF and RUL prediction step-by-step
print("Running Particle Filter and RUL prediction... this may take a moment.")
df_final = predict_RUL_series(df, pf_config)


# --- 4. Save and display results ---
# (Make sure 'results' directory exists)
df_final.to_csv('results/final_rul_prediction.csv', index=False)
print("Processing complete. Final results:")
print(df_final[['cycle', 'rho_filtered', 'D_filtered', 'A_t_filtered', 'alpha_t_filtered', 'RUL']].head())


# --- 5. Visualize all results ---
print("Generating plots...")
plot_all_results(df_final)
print("Done.")