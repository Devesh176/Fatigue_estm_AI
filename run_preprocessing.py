# run_preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re 

# Import our data-prep functions
from scripts.load_data import aggregate_coupon_data
from scripts.feature_extraction import compute_damage_indices
from scripts.create_labels import add_rul_label

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    print("Starting Data Preprocessing Pipeline...")
    
    data_root = Path("data")
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)
    
    layup_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("Layup")]
    
    if not layup_dirs:
        print(f"Error: No 'Layup*' folders found in {data_root}")
        return

    print(f"Found {len(layup_dirs)} layup(s).")
    
    all_coupon_data = [] # List to hold all processed dataframes

    for layup_dir in layup_dirs:
        coupon_dirs = [d for d in layup_dir.iterdir() if d.is_dir() and d.name.startswith("Coupon_")]
        
        print(f"  Processing {len(coupon_dirs)} coupon(s) in {layup_dir.name}...")
        
        for coupon_dir in coupon_dirs:
            coupon_name = coupon_dir.name
            
            match = re.search(r'(L\d+_S\d+)', coupon_name)
            if not match:
                print(f"    - Skipping {coupon_name}: Could not parse ID.")
                continue
            
            specimen_id = match.group(1)
            pzt_folder = coupon_dir / "PZT-data"
            strain_folder = coupon_dir / "StrainData"
            
            if not pzt_folder.exists() or not strain_folder.exists():
                print(f"    - Skipping {coupon_name}: Missing data folders.")
                continue
            
            print(f"    - Processing {coupon_name} (ID: {specimen_id})...")
            
            try:
                # 1. Load Data
                df_raw = aggregate_coupon_data(pzt_folder, strain_folder, specimen_id)
                if df_raw.empty:
                    print(f"    - Skipping {coupon_name}: No data loaded/aligned.")
                    continue

                # 2. Extract Features
                df_features = compute_damage_indices(df_raw)
                
                # 3. Create Label
                df_labeled = add_rul_label(df_features)

                # 4. Add a 'coupon_id' for tracking
                df_labeled['coupon_id'] = specimen_id
                
                # Clean and add to our master list
                df_clean = df_labeled.dropna()
                if not df_clean.empty:
                    all_coupon_data.append(df_clean)
                
                print(f"    - Success: Processed {len(df_clean)} data points.")

            except Exception as e:
                print(f"    - FAILED for {coupon_name}: {e}")

    # --- End of loop ---
    if not all_coupon_data:
        print("Pipeline finished, but no data was processed.")
        return
        
    # Combine all data into one big dataframe
    final_dataset = pd.concat(all_coupon_data, ignore_index=True)
    
    # Define the output path
    output_file = results_root / "ml_training_dataset.csv"
    final_dataset.to_csv(output_file, index=False)
    
    print("\nPipeline Finished.")
    print(f"Successfully saved all processed data to: {output_file}")
    print(f"Total data points: {len(final_dataset)}")
    print("\nDataset Head:")
    print(final_dataset.head())

if __name__ == "__main__":
    main()