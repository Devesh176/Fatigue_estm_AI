import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import sys

# --- Configuration ---
original_file = 'results/ml_training_dataset.csv'
output_file = 'results/synthetic_fatigue_data_adaptive.csv' # New output file name
# ---------------------

try:
    df = pd.read_csv(original_file)
except FileNotFoundError:
    print(f"Error: The file '{original_file}' was not found.")
    print("Please make sure it's in the same directory as this script.")
    sys.exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    sys.exit()

original_columns = df.columns
id_cols = ['coupon_id', 'cycle']
interp_cols = [col for col in df.columns if col not in id_cols]

unique_coupons = df['coupon_id'].unique()
synthetic_dfs = []

print(f"Starting ADAPTIVE synthetic data generation for {len(unique_coupons)} coupons...")

for coupon_id in unique_coupons:
    coupon_df = df[df['coupon_id'] == coupon_id].sort_values('cycle').reset_index(drop=True)
    
    n_points = len(coupon_df)

    # --- Adaptive Interpolation Kind Selection ---
    if n_points >= 4:
        kind = 'cubic'
    elif n_points == 3:
        kind = 'quadratic'
    elif n_points == 2:
        kind = 'linear'
    else:
        # Need at least 2 points to interpolate
        print(f"Skipping coupon_id {coupon_id}: needs at least 2 data points. Found: {n_points}")
        continue
    
    print(f"  -> Processing {coupon_id} ({n_points} points) using '{kind}' interpolation.")

    min_cycle = coupon_df['cycle'].min()
    max_cycle = coupon_df['cycle'].max()

    new_cycle_index = np.arange(min_cycle, max_cycle + 1)
    new_df = pd.DataFrame({'cycle': new_cycle_index})
    new_df['coupon_id'] = coupon_id

    # --- Apply the chosen interpolation ---
    for col in interp_cols:
        interp_func = interp1d(
            coupon_df['cycle'], # Original x-values
            coupon_df[col],     # Original y-values
            kind=kind,          # ADAPTIVE kind
            fill_value="extrapolate" 
        )
        
        new_df[col] = interp_func(new_cycle_index)

    # Reorder columns to match the original DataFrame
    new_df = new_df[original_columns]
    synthetic_dfs.append(new_df)

print("Concatenating all synthetic data...")
if synthetic_dfs:
    final_synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)
    final_synthetic_df.to_csv(output_file, index=False)

    print(f"\nSuccessfully generated and saved synthetic data to '{output_file}'.")
    print("\nSynthetic Data Info:")
    final_synthetic_df.info()
else:
    print("No data was generated. Check point requirements for interpolation.")