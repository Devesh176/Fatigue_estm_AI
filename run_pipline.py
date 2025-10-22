# run_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re # Import regex

# Import all our custom pipeline functions
from scripts.load_data import aggregate_coupon_data
from scripts.feature_extraction import compute_damage_indices
from scripts.rul_prediction import predict_RUL_series
from scripts.visualize_results import plot_all_results

# Suppress pandas FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    """
    Main orchestrator to run the RUL prediction pipeline
    for all layups and coupons.
    """
    print("Starting RUL Prediction Pipeline...")
    
    # --- 4. Particle Filter Configuration ---
    pf_config = {
        'N': 1000,
        'sigma_state_rho': 1e-3,
        'sigma_state_D': 1e-4,
        'sigma_param_At': 1e-7,
        'sigma_param_alpha_t': 1e-4,
        'init_state': [0.0, 1.0],
        'init_params_mean': [8.5e-4, 1.8],
        'init_params_cov': np.diag([1e-7, 0.1])
    }
    
    data_root = Path("data")
    results_root = Path("results")
    plots_root = Path("plots")
    
    results_root.mkdir(exist_ok=True)
    plots_root.mkdir(exist_ok=True)
    
    layup_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("Layup")]
    
    if not layup_dirs:
        print(f"Error: No 'Layup*' folders found in {data_root}")
        return

    print(f"Found {len(layup_dirs)} layup(s).")

    for layup_dir in layup_dirs:
        coupon_dirs = [d for d in layup_dir.iterdir() if d.is_dir() and d.name.startswith("Coupon_")]
        
        if not coupon_dirs:
            print(f"  No 'Coupon_*' folders found in {layup_dir.name}.")
            continue
            
        print(f"  Processing {len(coupon_dirs)} coupon(s) in {layup_dir.name}...")
        
        for coupon_dir in coupon_dirs:
            coupon_name = coupon_dir.name
            
            # --- START FIX ---
            # Dynamically find the Specimen ID (e.g., L1_S11) from the folder name
            match = re.search(r'(L\d+_S\d+)', coupon_name)
            if not match:
                print(f"    - Skipping {coupon_name}: Could not parse specimen ID from folder name.")
                continue
            
            specimen_id = match.group(1) # e.g., "L1_S11"
            
            # Define paths for BOTH folders
            pzt_folder = coupon_dir / "PZT-data"
            strain_folder = coupon_dir / "StrainData"
            
            if not pzt_folder.exists() or not strain_folder.exists():
                print(f"    - Skipping {coupon_name}: Missing 'PZT-data' or 'StrainData' folder.")
                continue
            # --- END FIX ---
            
            print(f"    - Processing {coupon_name} (ID: {specimen_id})...")
            
            try:
                # --- 2.2 Data Aggregation ---
                # --- START FIX ---
                # Pass all THREE required arguments
                df_raw = aggregate_coupon_data(pzt_folder, strain_folder, specimen_id)
                # --- END FIX ---

                if df_raw.empty:
                    print(f"    - Skipping {coupon_name}: No data loaded/aligned.")
                    continue

                # --- 2.3 Damage Feature Extraction ---
                df_damage = compute_damage_indices(df_raw)

                df_clean = df_damage.dropna(subset=['rho_smooth', 'D_smooth'])
                
                if df_clean.empty:
                    print(f"    - Skipping {coupon_name}: No valid data after cleaning.")
                    continue
                
                # --- 3-5. Run Filter & RUL Prediction ---
                df_final = predict_RUL_series(df_clean, pf_config)
                
                # --- 6. Visualization & Reporting ---
                csv_path = results_root / f"{coupon_name}_rul_prediction.csv"
                plot_path = plots_root / f"{coupon_name}_rul_plot.png"
                
                df_final.to_csv(csv_path, index=False)
                plot_all_results(df_final, coupon_name, plot_path)
                
                print(f"    - Success: Saved results to {csv_path} and {plot_path}")

            except Exception as e:
                print(f"    - FAILED for {coupon_name}: {e}")

    print("Pipeline finished.")

if __name__ == "__main__":
    main()