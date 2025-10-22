# scripts/1_load_data.py
import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path
import re

def get_pzt_amplitude(filepath):
    """Helper to load one PZT file and get mean amplitude."""
    try:
        mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        if 'coupon' not in mat or not hasattr(mat['coupon'], 'path_data'):
            return np.nan
            
        path_data = mat['coupon'].path_data
        amplitudes = []
        for p in np.atleast_1d(path_data):
            if hasattr(p, 'amplitude') and isinstance(p.amplitude, (int, float, np.number)):
                amplitudes.append(p.amplitude)
        if amplitudes:
            return np.mean(amplitudes)
    except Exception as e:
        print(f"  > Warning: Could not load PZT file {filepath.name}: {e}")
    return np.nan

def get_strain_median(filepath):
    """Helper to load one Strain file and get median strain."""
    try:
        mat = sio.loadmat(filepath)
        all_strains = []
        for field in ['strain1', 'strain2', 'strain3', 'strain4']:
            if field in mat:
                strain_array = mat[field].flatten()
                strain_array = strain_array[~np.isnan(strain_array)]
                if strain_array.size > 0:
                    all_strains.append(strain_array)
        
        if all_strains:
            combined = np.concatenate(all_strains)
            if combined.size > 0:
                return np.median(combined)
    except Exception as e:
        print(f"  > Warning: Could not load Strain file {filepath.name}: {e}")
    return np.nan

def aggregate_coupon_data(pzt_folder, strain_folder, specimen_id):
    """
    Loads PZT and Strain data from their separate folders and
    aligns them based on file names.
    """
    
    # --- 1. Load PZT Data ---
    # specimen_id is like "L1_S11"
    pzt_records = []
    pzt_id = specimen_id.replace('_', '') # L1_S11 -> L1S11
    
    # Regex matches L1S11_0_0.mat, L1S11_0_1_1.mat, L1S11_1000_1_2.mat
    pzt_regex = re.compile(rf'{pzt_id}_(\d+)_(\d+)(?:_(\d+))?\.mat')
    
    for f in pzt_folder.glob("*.mat"):
        match = pzt_regex.match(f.name)
        if match:
            cycle = int(match.group(1))
            condition = int(match.group(2))
            
            # Per your plan, we need Baseline (0) and Loaded (1)
            if condition in [0, 1]:
                pzt_records.append({
                    'cycle': cycle,
                    'condition': condition,
                    'mean_amplitude': get_pzt_amplitude(f)
                })
    
    if not pzt_records:
        print(f"  > Error: No valid PZT files (Condition 0 or 1) found for {pzt_id}.")
        return pd.DataFrame()
        
    df_pzt = pd.DataFrame(pzt_records)
    
    # We only want the "Loaded" data for damage, but "Baseline" for baseline
    # Let's average all repeats, keeping condition
    df_pzt = df_pzt.groupby(['cycle', 'condition']).mean().reset_index()
    
    # Separate baseline (condition 0) and loaded (condition 1)
    df_pzt_baseline = df_pzt[df_pzt['condition'] == 0].copy()
    df_pzt_loaded = df_pzt[df_pzt['condition'] == 1].copy()
    
    # Ensure baseline (cycle 0) is from condition 0 if available
    if not df_pzt_baseline.empty and 0 in df_pzt_baseline['cycle'].values:
        baseline_pzt = df_pzt_baseline[df_pzt_baseline['cycle'] == 0]
        # Combine with all loaded data (excluding cycle 0 if it's there)
        df_pzt = pd.concat([baseline_pzt, df_pzt_loaded[df_pzt_loaded['cycle'] != 0]])
    else:
        # No baseline file, just use the loaded files
        df_pzt = df_pzt_loaded
    
    df_pzt = df_pzt.sort_values('cycle').reset_index(drop=True)

    # --- 2. Load Strain Data ---
    strain_records = []
    
    # Regex to match F-files (e.g., L1_S11_F00_DAT.mat, L1_S11_F01_STRAIN_A_DAT.mat)
    strain_regex_f = re.compile(rf'{specimen_id}_F(\d+)_.*\.mat')
    # Regex to match S-files (e.g., L1_S11_S00_DAT.mat)
    strain_regex_s = re.compile(rf'{specimen_id}_S(\d+)_.*\.mat')
    
    for f in strain_folder.glob("*.mat"):
        match_f = strain_regex_f.match(f.name)
        match_s = strain_regex_s.match(f.name)
        
        index = -1
        is_baseline = False
        
        if match_f:
            index = int(match_f.group(1)) # F-index: 0, 1, 2, ...
            if index == 0:
                is_baseline = True
        elif match_s:
            index = int(match_s.group(1)) # S-index: 0, 8, 10
            if index == 0:
                is_baseline = True

        if index != -1:
            strain_records.append({
                'index': index,
                'is_baseline': is_baseline,
                'median_strain': get_strain_median(f)
            })

    if not strain_records:
        print(f"  > Error: No valid Strain files matching F* or S* found for {specimen_id}.")
        return pd.DataFrame()
        
    df_strain = pd.DataFrame(strain_records)
    # Average all readings (A, M, S, DAT) for each index
    df_strain = df_strain.groupby('index').mean().reset_index()

    # Separate baseline (index 0) from fatigue files
    df_strain_baseline = df_strain[(df_strain['is_baseline'] == True) & (df_strain['index'] == 0)]
    df_strain_fatigue = df_strain[df_strain['is_baseline'] == False].sort_values('index')

    if df_strain_baseline.empty:
         print(f"  > Warning: No baseline (F00 or S00) strain file found for {specimen_id}.")
         # Fallback: use the first fatigue file as baseline
         if not df_strain_fatigue.empty:
             df_strain_baseline = df_strain_fatigue.iloc[0:1]
             df_strain_fatigue = df_strain_fatigue.iloc[1:]
         else:
             print("  > Error: No strain data to use.")
             return pd.DataFrame()
    
    # Combine baseline + fatigue sequence
    df_strain = pd.concat([df_strain_baseline, df_strain_fatigue]).reset_index(drop=True)

    # --- 3. Align and Merge Data ---
    # This is the new, robust alignment logic.
    # We align by *order*, not by a shared key.
    
    if len(df_pzt) != len(df_strain):
        print(f"  > Warning: Mismatch in PZT/Strain file counts. Aligning based on order.")
        print(f"  > Found {len(df_pzt)} PZT data points (Cycles: {df_pzt['cycle'].tolist()})")
        print(f"  > Found {len(df_strain)} Strain data points (Indices: {df_strain['index'].tolist()})")
        
        # Truncate to the shorter list
        min_len = min(len(df_pzt), len(df_strain))
        df_pzt = df_pzt.iloc[:min_len]
        df_strain = df_strain.iloc[:min_len]
        
    # Reset index on both to ensure a clean 0-N index
    df_pzt = df_pzt.reset_index(drop=True)
    df_strain = df_strain.reset_index(drop=True)

    # Combine the dataframes side-by-side
    df_merged = pd.concat([df_pzt, df_strain['median_strain']], axis=1)
    
    # Final cleanup
    df_merged = df_merged.dropna(subset=['mean_amplitude', 'median_strain'])
    
    if df_merged.empty:
        print(f"  > Error: Data alignment failed for {specimen_id}.")
    
    return df_merged[['cycle', 'mean_amplitude', 'median_strain']]