# # scripts/1_load_data.py
# import scipy.io as sio
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import re

# def get_pzt_amplitude(filepath):
#     """Helper to load one PZT file and get mean amplitude."""
#     try:
#         mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
#         if 'coupon' not in mat or not hasattr(mat['coupon'], 'path_data'):
#             return np.nan
            
#         path_data = mat['coupon'].path_data
#         amplitudes = []
#         for p in np.atleast_1d(path_data):
#             if hasattr(p, 'amplitude') and isinstance(p.amplitude, (int, float, np.number)):
#                 amplitudes.append(p.amplitude)
#         if amplitudes:
#             return np.mean(amplitudes)
#     except Exception as e:
#         print(f"  > Warning: Could not load PZT file {filepath.name}: {e}")
#     return np.nan

# def get_strain_median(filepath):
#     """Helper to load one Strain file and get median strain."""
#     try:
#         mat = sio.loadmat(filepath)
#         all_strains = []
#         for field in ['strain1', 'strain2', 'strain3', 'strain4']:
#             if field in mat:
#                 strain_array = mat[field].flatten()
#                 strain_array = strain_array[~np.isnan(strain_array)]
#                 if strain_array.size > 0:
#                     all_strains.append(strain_array)
        
#         if all_strains:
#             combined = np.concatenate(all_strains)
#             if combined.size > 0:
#                 return np.median(combined)
#     except Exception as e:
#         print(f"  > Warning: Could not load Strain file {filepath.name}: {e}")
#     return np.nan

# def aggregate_coupon_data(pzt_folder, strain_folder, specimen_id):
#     """
#     Loads PZT and Strain data from their separate folders and
#     aligns them based on file names.
#     """
    
#     # --- 1. Load PZT Data ---
#     # specimen_id is like "L1_S11"
#     pzt_records = []
#     pzt_id = specimen_id.replace('_', '') # L1_S11 -> L1S11
    
#     # Regex matches L1S11_0_0.mat, L1S11_0_1_1.mat, L1S11_1000_1_2.mat
#     pzt_regex = re.compile(rf'{pzt_id}_(\d+)_(\d+)(?:_(\d+))?\.mat')
    
#     for f in pzt_folder.glob("*.mat"):
#         match = pzt_regex.match(f.name)
#         if match:
#             cycle = int(match.group(1))
#             condition = int(match.group(2))
            
#             # Per your plan, we need Baseline (0) and Loaded (1)
#             if condition in [0, 1]:
#                 pzt_records.append({
#                     'cycle': cycle,
#                     'condition': condition,
#                     'mean_amplitude': get_pzt_amplitude(f)
#                 })
    
#     if not pzt_records:
#         print(f"  > Error: No valid PZT files (Condition 0 or 1) found for {pzt_id}.")
#         return pd.DataFrame()
        
#     df_pzt = pd.DataFrame(pzt_records)
    
#     # We only want the "Loaded" data for damage, but "Baseline" for baseline
#     # Let's average all repeats, keeping condition
#     df_pzt = df_pzt.groupby(['cycle', 'condition']).mean().reset_index()
    
#     # Separate baseline (condition 0) and loaded (condition 1)
#     df_pzt_baseline = df_pzt[df_pzt['condition'] == 0].copy()
#     df_pzt_loaded = df_pzt[df_pzt['condition'] == 1].copy()
    
#     # Ensure baseline (cycle 0) is from condition 0 if available
#     if not df_pzt_baseline.empty and 0 in df_pzt_baseline['cycle'].values:
#         baseline_pzt = df_pzt_baseline[df_pzt_baseline['cycle'] == 0]
#         # Combine with all loaded data (excluding cycle 0 if it's there)
#         df_pzt = pd.concat([baseline_pzt, df_pzt_loaded[df_pzt_loaded['cycle'] != 0]])
#     else:
#         # No baseline file, just use the loaded files
#         df_pzt = df_pzt_loaded
    
#     df_pzt = df_pzt.sort_values('cycle').reset_index(drop=True)

#     # --- 2. Load Strain Data ---
#     strain_records = []
    
#     # Regex to match F-files (e.g., L1_S11_F00_DAT.mat, L1_S11_F01_STRAIN_A_DAT.mat)
#     strain_regex_f = re.compile(rf'{specimen_id}_F(\d+)_.*\.mat')
#     # Regex to match S-files (e.g., L1_S11_S00_DAT.mat)
#     strain_regex_s = re.compile(rf'{specimen_id}_S(\d+)_.*\.mat')
    
#     for f in strain_folder.glob("*.mat"):
#         match_f = strain_regex_f.match(f.name)
#         match_s = strain_regex_s.match(f.name)
        
#         index = -1
#         is_baseline = False
        
#         if match_f:
#             index = int(match_f.group(1)) # F-index: 0, 1, 2, ...
#             if index == 0:
#                 is_baseline = True
#         elif match_s:
#             index = int(match_s.group(1)) # S-index: 0, 8, 10
#             if index == 0:
#                 is_baseline = True

#         if index != -1:
#             strain_records.append({
#                 'index': index,
#                 'is_baseline': is_baseline,
#                 'median_strain': get_strain_median(f)
#             })

#     if not strain_records:
#         print(f"  > Error: No valid Strain files matching F* or S* found for {specimen_id}.")
#         return pd.DataFrame()
        
#     df_strain = pd.DataFrame(strain_records)
#     # Average all readings (A, M, S, DAT) for each index
#     df_strain = df_strain.groupby('index').mean().reset_index()

#     # Separate baseline (index 0) from fatigue files
#     df_strain_baseline = df_strain[(df_strain['is_baseline'] == True) & (df_strain['index'] == 0)]
#     df_strain_fatigue = df_strain[df_strain['is_baseline'] == False].sort_values('index')

#     if df_strain_baseline.empty:
#          print(f"  > Warning: No baseline (F00 or S00) strain file found for {specimen_id}.")
#          # Fallback: use the first fatigue file as baseline
#          if not df_strain_fatigue.empty:
#              df_strain_baseline = df_strain_fatigue.iloc[0:1]
#              df_strain_fatigue = df_strain_fatigue.iloc[1:]
#          else:
#              print("  > Error: No strain data to use.")
#              return pd.DataFrame()
    
#     # Combine baseline + fatigue sequence
#     df_strain = pd.concat([df_strain_baseline, df_strain_fatigue]).reset_index(drop=True)

#     # --- 3. Align and Merge Data ---
#     # This is the new, robust alignment logic.
#     # We align by *order*, not by a shared key.
    
#     if len(df_pzt) != len(df_strain):
#         print(f"  > Warning: Mismatch in PZT/Strain file counts. Aligning based on order.")
#         print(f"  > Found {len(df_pzt)} PZT data points (Cycles: {df_pzt['cycle'].tolist()})")
#         print(f"  > Found {len(df_strain)} Strain data points (Indices: {df_strain['index'].tolist()})")
        
#         # Truncate to the shorter list
#         min_len = min(len(df_pzt), len(df_strain))
#         df_pzt = df_pzt.iloc[:min_len]
#         df_strain = df_strain.iloc[:min_len]
        
#     # Reset index on both to ensure a clean 0-N index
#     df_pzt = df_pzt.reset_index(drop=True)
#     df_strain = df_strain.reset_index(drop=True)

#     # Combine the dataframes side-by-side
#     df_merged = pd.concat([df_pzt, df_strain['median_strain']], axis=1)
    
#     # Final cleanup
#     df_merged = df_merged.dropna(subset=['mean_amplitude', 'median_strain'])
    
#     if df_merged.empty:
#         print(f"  > Error: Data alignment failed for {specimen_id}.")
    
#     return df_merged[['cycle', 'mean_amplitude', 'median_strain']]

# scripts/1_load_data.py
import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path
import re

def get_pzt_amplitude(filepath):
    """Helper to load one PZT file and get mean amplitude from sensor signals."""
    try:
        # Load regardless of extension, assume it's a mat file if readable
        mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        if 'coupon' not in mat or not hasattr(mat['coupon'], 'path_data'):
            return np.nan
            
        path_data = mat['coupon'].path_data
        path_amplitudes = [] 
        
        for p in np.atleast_1d(path_data):
            if hasattr(p, 'signal_sensor'):
                signal = p.signal_sensor
                if isinstance(signal, np.ndarray) and np.issubdtype(signal.dtype, np.number):
                    if signal.size > 0:
                        path_amp = np.max(np.abs(signal))
                        path_amplitudes.append(path_amp)

        if path_amplitudes:
            return np.mean(path_amplitudes) 
            
    except Exception as e:
        print(f"  > Warning: Could not load PZT file {filepath.name}: {e}")
    return np.nan

def get_strain_median(filepath):
    """Helper to load one Strain file and get median of gauge means."""
    try:
        # Load regardless of extension
        mat = sio.loadmat(filepath)
        gauge_means = []
        for field in ['strain1', 'strain2', 'strain3', 'strain4']:
            if field in mat:
                strain_signal = mat[field].flatten()
                strain_signal = strain_signal[~np.isnan(strain_signal)]
                if strain_signal.size > 0:
                    gauge_means.append(np.mean(strain_signal))
        
        if gauge_means:
            return np.median(gauge_means)
            
    except Exception as e:
        print(f"  > Warning: Could not load Strain file {filepath.name}: {e}")
    return np.nan

def aggregate_coupon_data(pzt_folder, strain_folder, specimen_id):
    """
    Loads PZT and Strain data by reading the "link" inside
    the PZT .mat file, using flexible regex for strain files (with or without extension).
    """
    
    # --- 1. Load ALL Strain Data into a map ---
    print(f"  > Building Strain data map for {specimen_id}...")
    strain_map = {}
    
    # --- START FIX: Flexible Regex and Glob ---
    # Matches L2_S17_1_F001 (no extension)
    # Allows for optional .mat extension
    strain_regex = re.compile(rf'{specimen_id}_\d+_([FS]\d+)(?:\.mat)?$') 
    
    # Search for files with or without .mat
    for f in strain_folder.glob(f"{specimen_id}_*"): 
    # --- END FIX ---
        match = strain_regex.match(f.name)
        if match:
            # The key is the F/S index, e.g., "F01", "S001"
            # Note: Your example S001 has 3 digits, F001 has 3. Let's keep it flexible.
            index_key = match.group(1) 
            median_strain = get_strain_median(f)
            if not np.isnan(median_strain):
                # If key already exists (e.g., A/M/S for same index), average
                if index_key in strain_map:
                    strain_map[index_key] = (strain_map[index_key] + median_strain) / 2
                else:
                    strain_map[index_key] = median_strain

    if not strain_map:
        print(f"  > Error: No valid Strain files found for {specimen_id}.")
        return pd.DataFrame()

    # --- 2. Load PZT Data and Align ---
    pzt_records = []
    pzt_id = specimen_id.replace('_', '') 
    pzt_regex = re.compile(rf'{pzt_id}_(\d+)_(\d+)(?:_(\d+))?\.mat')
    
    for f in pzt_folder.glob("*.mat"):
        match = pzt_regex.match(f.name)
        if match:
            cycle = int(match.group(1))
            condition = int(match.group(2))
            
            if condition in [0, 1]: # Baseline (0) or Loaded (1)
                try:
                    mat = sio.loadmat(f, struct_as_record=False, squeeze_me=True)
                    if 'coupon' not in mat: continue
                    
                    coupon = mat['coupon']
                    mean_amp = get_pzt_amplitude(f)
                    
                    strain_key = None
                    median_strain = np.nan
                    
                    if hasattr(coupon, 'straingage_data') and hasattr(coupon.straingage_data, 'file_location'):
                        file_loc = coupon.straingage_data.file_location
                        
                        if isinstance(file_loc, str):
                            strain_key = file_loc 
                        elif isinstance(file_loc, np.ndarray):
                            strain_key = "".join(file_loc)

                    if strain_key:
                        # Extract the F/S index from the PZT file's link
                        # Regex now looks for Fxxx or Sxxx at the end
                        key_match = re.search(r'([FS]\d+)$', strain_key) 
                        if key_match:
                            strain_map_key = key_match.group(1) # e.g., "F004", "S001"
                            median_strain = strain_map.get(strain_map_key, np.nan)
                    
                    # Fallback for baseline (cycle 0)
                    if cycle == 0 and np.isnan(median_strain):
                        # Try S001 first based on your debug output
                        median_strain = strain_map.get('S001', strain_map.get('F00', np.nan)) 

                    # Only add record if we found amplitude
                    if not np.isnan(mean_amp):
                        pzt_records.append({
                            'cycle': cycle,
                            'condition': condition,
                            'mean_amplitude': mean_amp,
                            'median_strain': median_strain # Might be NaN here
                        })
                except Exception as e:
                    # Catch specific sio errors if files are not matlab files
                    if isinstance(e, (sio.matlab.miobase.MatReadError, TypeError)):
                         print(f"  > Warning: Skipping non-MATLAB file? {f.name}: {e}")
                    else:
                         print(f"  > Warning: Error reading {f.name}: {e}")


    if not pzt_records:
        print(f"  > Error: No valid PZT files (Condition 0 or 1) found for {pzt_id}.")
        return pd.DataFrame()
        
    df = pd.DataFrame(pzt_records)
    df = df[df['condition'].isin([0, 1])].copy()
    
    if df.empty:
        print(f"  > Error: No 'Loaded' (1) or 'Baseline' (0) files found after filtering.")
        return pd.DataFrame()
        
    # Average repeats for the same cycle
    df = df.groupby('cycle').mean().reset_index()
    
    # Fill NaNs - Crucial step
    df['mean_amplitude'] = df['mean_amplitude'].ffill().bfill()
    df['median_strain'] = df['median_strain'].ffill().bfill()
    
    # Drop rows ONLY IF BOTH are NaN AFTER filling
    # Or if median_strain is still NaN (means alignment failed)
    df = df.dropna(subset=['mean_amplitude', 'median_strain'], how='any')

    if df.empty:
        print(f"  > Error: No rows remaining after NaN handling for {specimen_id}.")
        
    return df