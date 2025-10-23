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
# ========================================================================================================#

# # scripts/1_load_data.py
# import scipy.io as sio
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import re

# def get_pzt_data(filepath):
#     """
#     Helper to load one PZT file and get mean amplitude and load.
#     Returns: tuple (mean_amplitude, load_value)
#     """
#     mean_amplitude = np.nan
#     load_value = np.nan
#     try:
#         mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
#         if 'coupon' not in mat:
#             return mean_amplitude, load_value # Return NaNs if no coupon struct

#         coupon = mat['coupon']

#         # Get PZT amplitude
#         if hasattr(coupon, 'path_data'):
#             path_data = coupon.path_data
#             amplitudes = []
#             for p in np.atleast_1d(path_data):
#                 if hasattr(p, 'signal_sensor'):
#                     signal = p.signal_sensor
#                     if isinstance(signal, np.ndarray) and np.issubdtype(signal.dtype, np.number):
#                         if signal.size > 0:
#                             path_amp = np.max(np.abs(signal))
#                             amplitudes.append(path_amp)
#             if amplitudes:
#                 mean_amplitude = np.mean(amplitudes)

#         # Get Load value
#         if hasattr(coupon, 'load') and isinstance(coupon.load, (int, float, np.number)):
#             load_value = float(coupon.load) # Ensure it's a float

#     except Exception as e:
#         print(f"  > Warning: Could not load PZT file {filepath.name}: {e}")

#     return mean_amplitude, load_value # Return potentially NaNs


# def get_strain_median(filepath):
#     """Helper to load one Strain file and get median of gauge means."""
#     try:
#         mat = sio.loadmat(filepath)
#         gauge_means = []
#         for field in ['strain1', 'strain2', 'strain3', 'strain4']:
#             if field in mat:
#                 strain_signal = mat[field].flatten()
#                 strain_signal = strain_signal[~np.isnan(strain_signal)]
#                 if strain_signal.size > 0:
#                     gauge_means.append(np.mean(strain_signal))

#         if gauge_means:
#             return np.median(gauge_means)

#     except Exception as e:
#         print(f"  > Warning: Could not load Strain file {filepath.name}: {e}")
#     return np.nan

# def aggregate_coupon_data(pzt_folder, strain_folder, specimen_id):
#     """
#     Loads PZT and Strain data from their separate folders and
#     aligns them using dynamically built regexes based on specimen_id
#     and F/S indices. Includes 'load'.
#     """

#     # specimen_id is like "L1_S11"

#     # --- 1. Load PZT Data (including Load) ---
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
#                 mean_amp, load_val = get_pzt_data(f) # Get both values
#                 pzt_records.append({
#                     'cycle': cycle,
#                     'condition': condition,
#                     'mean_amplitude': mean_amp,
#                     'load': load_val # Add load here
#                 })

#     if not pzt_records:
#         print(f"  > Error: No valid PZT files (Condition 0 or 1) found for {pzt_id}.")
#         return pd.DataFrame()

#     df_pzt = pd.DataFrame(pzt_records)

#     # Average repeats, keeping condition and load
#     # Use 'first' for load, assuming it's constant for repeats of the same cycle/condition
#     agg_funcs_pzt = {'mean_amplitude': 'mean', 'load': 'first'}
#     df_pzt = df_pzt.groupby(['cycle', 'condition']).agg(agg_funcs_pzt).reset_index()


#     # Separate baseline (condition 0) and loaded (condition 1)
#     df_pzt_baseline = df_pzt[df_pzt['condition'] == 0].copy()
#     df_pzt_loaded = df_pzt[df_pzt['condition'] == 1].copy()

#     # Ensure baseline (cycle 0) is from condition 0 if available
#     baseline_pzt_rec = None
#     if not df_pzt_baseline.empty and 0 in df_pzt_baseline['cycle'].values:
#         baseline_pzt_rec = df_pzt_baseline[df_pzt_baseline['cycle'] == 0]
#         df_pzt = pd.concat([baseline_pzt_rec, df_pzt_loaded[df_pzt_loaded['cycle'] != 0]])
#     else:
#         if 0 in df_pzt_loaded['cycle'].values:
#              baseline_pzt_rec = df_pzt_loaded[df_pzt_loaded['cycle'] == 0]
#              df_pzt = df_pzt_loaded
#         else:
#              print(f"  > Warning: No baseline (Cycle 0) PZT file found for {pzt_id}.")
#              df_pzt = df_pzt_loaded

#     df_pzt = df_pzt.sort_values('cycle').reset_index(drop=True)


#     # --- 2. Load Strain Data ---
#     strain_records = []

#     strain_regex_f = re.compile(rf'{specimen_id}_F(\d+).*?(?:\.mat)?$')
#     strain_regex_s = re.compile(rf'{specimen_id}_S(\d+).*?(?:\.mat)?$')

#     for f in strain_folder.glob(f"{specimen_id}_*"): # Search flexibly
#         match_f = strain_regex_f.match(f.name)
#         match_s = strain_regex_s.match(f.name)

#         index = -1
#         is_baseline = False

#         if match_f:
#             index = int(match_f.group(1)) # F-index: 0, 1, 2, ...
#             if index == 0:
#                 is_baseline = True
#         elif match_s:
#             s_index_str = match_s.group(1)
#             index = int(s_index_str) # S-index: 0, 1, 8, ...
#             # Be more flexible with baseline S-indices
#             if s_index_str in ['0', '00', '000', '001', '1']:
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
#     df_strain_baseline = df_strain[df_strain['is_baseline'] == True]
#     df_strain_fatigue = df_strain[df_strain['is_baseline'] == False].sort_values('index')

#     if df_strain_baseline.empty:
#          print(f"  > Warning: No baseline (F00 or S00/S01) strain file found for {specimen_id}.")
#          if not df_strain_fatigue.empty:
#              df_strain_baseline = df_strain_fatigue.iloc[0:1].copy()
#              df_strain_baseline['is_baseline'] = True
#              df_strain_fatigue = df_strain_fatigue.iloc[1:]
#          else:
#              print("  > Error: No strain data to use.")
#              return pd.DataFrame()

#     # Ensure baseline index is 0
#     df_strain_baseline['index'] = 0

#     df_strain = pd.concat([df_strain_baseline, df_strain_fatigue]).reset_index(drop=True)

#     # --- 3. Align and Merge Data ---
#     if len(df_pzt) != len(df_strain):
#         print(f"  > Warning: Mismatch in PZT/Strain file counts. Aligning based on order.")
#         print(f"  > Found {len(df_pzt)} PZT data points (Cycles: {df_pzt['cycle'].tolist()})")
#         print(f"  > Found {len(df_strain)} Strain data points (Indices: {df_strain['index'].tolist()})")
#         min_len = min(len(df_pzt), len(df_strain))
#         df_pzt = df_pzt.iloc[:min_len]
#         df_strain = df_strain.iloc[:min_len]

#     df_pzt = df_pzt.reset_index(drop=True)
#     df_strain = df_strain.reset_index(drop=True)

#     # Combine side-by-side, now including 'load'
#     df_merged = pd.concat([df_pzt[['cycle', 'mean_amplitude', 'load']], df_strain['median_strain']], axis=1)

#     # --- 4. Final Cleanup ---
#     # Fill NaNs before dropping
#     df_merged['mean_amplitude'] = df_merged['mean_amplitude'].ffill().bfill()
#     df_merged['median_strain'] = df_merged['median_strain'].ffill().bfill()
#     df_merged['load'] = df_merged['load'].ffill().bfill() # Fill load NaNs

#     # Drop rows if critical data is still missing
#     df_merged = df_merged.dropna(subset=['mean_amplitude', 'median_strain', 'load'])


#     if df_merged.empty:
#         print(f"  > Error: Data alignment failed or critical data missing for {specimen_id}.")

#     return df_merged[['cycle', 'mean_amplitude', 'median_strain', 'load']] # Return load

# scripts/1_load_data.py
import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path
import re
import os

# --- START MODIFICATION: Enhanced PZT Feature Extraction ---
def get_pzt_data(filepath):
    """
    Loads PZT file, returns dictionary of statistics:
    mean_amplitude, load, pzt_amplitude_std, pzt_energy_mean, pzt_rms_mean
    """
    stats = {
        'mean_amplitude': np.nan, 'load': np.nan,
        'pzt_amplitude_std': np.nan, 'pzt_energy_mean': np.nan,
        'pzt_rms_mean': np.nan
    }
    try:
        mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        if 'coupon' not in mat: return stats
        coupon = mat['coupon']

        # Get PZT Signal Stats
        if hasattr(coupon, 'path_data'):
            path_data = coupon.path_data
            path_amplitudes = []
            path_energies = []
            path_rms_values = []

            for p in np.atleast_1d(path_data):
                if hasattr(p, 'signal_sensor'):
                    signal = p.signal_sensor
                    if isinstance(signal, np.ndarray) and np.issubdtype(signal.dtype, np.number):
                        if signal.size > 0:
                            # Amplitude (Max Abs)
                            path_amp = np.max(np.abs(signal))
                            path_amplitudes.append(path_amp)
                            # Energy (Sum of Squares)
                            path_energies.append(np.sum(signal**2))
                            # RMS (Root Mean Square of signal)
                            path_rms_values.append(np.sqrt(np.mean(signal**2)))

            # Calculate stats across paths
            if path_amplitudes:
                stats['mean_amplitude'] = np.mean(path_amplitudes)
                stats['pzt_amplitude_std'] = np.std(path_amplitudes)
            if path_energies:
                stats['pzt_energy_mean'] = np.mean(path_energies)
            if path_rms_values:
                 stats['pzt_rms_mean'] = np.mean(path_rms_values)

        # Get Load value
        if hasattr(coupon, 'load') and isinstance(coupon.load, (int, float, np.number)):
            stats['load'] = float(coupon.load)

    except Exception as e:
        if isinstance(e, (sio.matlab.miobase.MatReadError, TypeError, ValueError)): pass
        else: print(f"  > Warning: Could not load PZT file {filepath.name}: {e}")
    return stats
# --- END MODIFICATION ---

# --- START MODIFICATION: Enhanced Strain Feature Extraction ---
def get_strain_stats(filepath):
    """
    Loads Strain file, returns dictionary of statistics:
    median_strain (median of means), strain_mean_of_means, strain_std_of_means
    """
    stats = {
        'median_strain': np.nan,
        'strain_mean_of_means': np.nan,
        'strain_std_of_means': np.nan
    }
    try:
        mat = sio.loadmat(filepath)
        gauge_means = [] # Store mean value of each gauge signal
        for field in ['strain1', 'strain2', 'strain3', 'strain4']:
            if field in mat:
                strain_signal = mat[field].flatten()
                strain_signal = strain_signal[~np.isnan(strain_signal)]
                if strain_signal.size > 0:
                    gauge_means.append(np.mean(strain_signal))

        if gauge_means:
            stats['median_strain'] = np.median(gauge_means)
            stats['strain_mean_of_means'] = np.mean(gauge_means)
            if len(gauge_means) > 1: # Need at least 2 gauges for std dev
                stats['strain_std_of_means'] = np.std(gauge_means)
            else:
                 stats['strain_std_of_means'] = 0.0 # Std dev is 0 if only one gauge


    except Exception as e:
         if isinstance(e, (sio.matlab.miobase.MatReadError, TypeError, ValueError)): pass
         elif 'is a directory' not in str(e): print(f"  > Warning: Could not load Strain file {filepath.name}: {e}")
    return stats
# --- END MODIFICATION ---


def find_logbook_file(coupon_dir, specimen_id):
    """Finds the LogBook file (e.g., L1S12.xlsx or L1S12.csv)."""
    base_name = specimen_id.replace('_', '')
    xlsx_file = coupon_dir / f"{base_name}.xlsx"
    if xlsx_file.exists(): return xlsx_file
    csv_file = coupon_dir / f"{base_name}.csv"
    if csv_file.exists(): return csv_file
    for ext in ['.xlsx', '.csv']:
        log_files = list(coupon_dir.glob(f"*[Ll][Oo][Gg]*{ext}"))
        if log_files:
            print(f"  > Warning: Using fallback LogBook name: {log_files[0].name}")
            return log_files[0]
    return None

def find_file_robustly(folder_path, filename_from_log):
    """Tries to find a file even with extension/suffix missing, case differences, or quotes."""
    if pd.isna(filename_from_log) or not filename_from_log: return None
    filename_from_log = str(filename_from_log).strip().strip("'").strip('"')
    exact_path = folder_path / filename_from_log
    if exact_path.is_file(): return exact_path
    if not filename_from_log.lower().endswith('.mat'):
        path_with_ext = folder_path / f"{filename_from_log}.mat"
        if path_with_ext.is_file(): return path_with_ext
    core_match = re.match(r'(L\d+_S\d+_[FS]\d+)', filename_from_log, re.IGNORECASE)
    core_name_log = core_match.group(1).lower() if core_match else None
    pzt_core_match = re.match(r'(L\d+S\d+_\d+_\d+)', filename_from_log.replace('_',''), re.IGNORECASE)
    pzt_core_name_log = pzt_core_match.group(1).lower() if pzt_core_match else None
    try:
        for item in folder_path.iterdir():
            if not item.is_file(): continue
            item_stem_lower = item.stem.lower()
            log_stem_lower = Path(filename_from_log).stem.lower()
            if item_stem_lower == log_stem_lower:
                if item.suffix.lower() in ['.mat', '']: return item
            if core_name_log:
                 item_core_match = re.match(r'(L\d+_S\d+_[FS]\d+)', item_stem_lower, re.IGNORECASE)
                 item_core_name = item_core_match.group(1) if item_core_match else None
                 if item_core_name == core_name_log:
                      if item.suffix.lower() in ['.mat', '']: return item
            if pzt_core_name_log:
                 item_pzt_core_match = re.match(r'(L\d+S\d+_\d+_\d+)', item_stem_lower.replace('_',''), re.IGNORECASE)
                 item_pzt_core_name = item_pzt_core_match.group(1) if item_pzt_core_match else None
                 if item_pzt_core_name == pzt_core_name_log:
                     if item.suffix.lower() in ['.mat', '']: return item
    except FileNotFoundError: pass
    if '.' in filename_from_log:
        path_no_ext = folder_path / Path(filename_from_log).stem
        if path_no_ext.is_file(): return path_no_ext
    return None


def aggregate_coupon_data(pzt_folder, strain_folder, specimen_id):
    """
    Loads PZT and Strain data guided by the LogBook file, using direct
    filename pairing and robust file searching. Includes additional features.
    """
    coupon_dir = pzt_folder.parent
    logbook_file = find_logbook_file(coupon_dir, specimen_id)
    if not logbook_file:
        print(f"  > Error: LogBook file not found in {coupon_dir}.")
        return pd.DataFrame()
    print(f"  > Using LogBook: {logbook_file.name}")

    # --- Read LogBook ---
    try:
        header_row=0
        if logbook_file.suffix == '.csv':
            try:
                peek_df = pd.read_csv(logbook_file, nrows=5, encoding_errors='ignore')
                if all('unnamed:' in str(col).lower() for col in peek_df.columns):
                    header_row = None
                    for skip in [1, 2]:
                        try:
                            log_df = pd.read_csv(logbook_file, skiprows=skip, encoding_errors='ignore')
                            if not all('unnamed:' in str(col).lower() for col in log_df.columns):
                                header_row = skip; print(f"  > Detected header row at line {header_row + 1} in CSV."); break
                        except Exception: continue
                    else: header_row = None; print("  > Warning: Could not detect header in CSV.")
            except Exception: header_row = None
            log_df = pd.read_csv(logbook_file, header=header_row, encoding_errors='ignore')
        else: # XLSX
             try:
                log_df = pd.read_excel(logbook_file, header=header_row)
                if all('unnamed:' in str(col).lower() for col in log_df.columns): log_df = pd.read_excel(logbook_file, header=1)
                if all('unnamed:' in str(col).lower() for col in log_df.columns): log_df = pd.read_excel(logbook_file, header=2)
                if all('unnamed:' in str(col).lower() for col in log_df.columns): print("  > Error: Could not detect header in Excel."); return pd.DataFrame()
             except Exception as read_err: print(f"  > Error: Failed to read Excel: {read_err}"); return pd.DataFrame()
    except Exception as e: print(f"  > Error: Could not read LogBook file {logbook_file.name}: {e}"); return pd.DataFrame()

    # --- Standardize Columns & Identify ---
    log_df.columns = log_df.columns.str.strip().str.lower()
    cycle_col = next((col for col in log_df.columns if 'cycle' in col), None)
    pzt_col = next((col for col in log_df.columns if ('pzt' in col or 'lamb' in col or 'data file' in col) and 'condition 1' in col), None)
    if not pzt_col: pzt_col = next((col for col in log_df.columns if 'data file name' in col), None)
    strain_col = next((col for col in log_df.columns if 'mts file' in col), None)
    if not strain_col: strain_col = next((col for col in log_df.columns if 'strain' in col and ('index' in col or 'file' in col)), None)
    load_col = next((col for col in log_df.columns if 'load' in col), None)
    required_cols = [cycle_col, pzt_col, strain_col]
    if not all(required_cols):
        print(f"  > Error: Could not identify all required LogBook columns."); return pd.DataFrame()

    # --- Clean LogBook Data ---
    log_df = log_df.dropna(subset=[cycle_col])
    log_df[cycle_col] = pd.to_numeric(log_df[cycle_col], errors='coerce')
    log_df = log_df.dropna(subset=[cycle_col])
    log_df[cycle_col] = log_df[cycle_col].astype(int)
    log_df[pzt_col] = log_df[pzt_col].fillna('').astype(str).str.strip()
    log_df[strain_col] = log_df[strain_col].fillna('').astype(str).str.strip()

    # --- Iterate through LogBook to build final DataFrame ---
    aligned_records = []
    processed_pzt_files = set()
    print(f"  > Processing {len(log_df)} rows from LogBook...")

    for _, row in log_df.iterrows():
        cycle = row[cycle_col]
        pzt_filename_log = row[pzt_col]
        strain_filenames_log_str = row[strain_col]

        if not pzt_filename_log or pzt_filename_log in processed_pzt_files: continue

        pzt_filepath = find_file_robustly(pzt_folder, pzt_filename_log)
        if not pzt_filepath: continue

        # --- START MODIFICATION: Process Strain Stats ---
        strain_stats_list = [] # Store stats dict from each found file
        if strain_filenames_log_str:
            potential_strain_files = strain_filenames_log_str.split()
            for strain_filename_part in potential_strain_files:
                strain_filepath = find_file_robustly(strain_folder, strain_filename_part)
                if strain_filepath:
                    # Get the dictionary of strain stats
                    strain_stats_dict = get_strain_stats(strain_filepath)
                    # Only add if we got a valid median_strain
                    if not np.isnan(strain_stats_dict['median_strain']):
                        strain_stats_list.append(strain_stats_dict)

        # Average the stats if multiple strain files were found and loaded
        final_strain_stats = {
            'median_strain': np.nan,
            'strain_mean_of_means': np.nan,
            'strain_std_of_means': np.nan
        }
        if strain_stats_list:
            temp_df = pd.DataFrame(strain_stats_list)
            # Use mean to average stats across files (A/M/S) for the same cycle
            averaged_stats = temp_df.mean().to_dict()
            final_strain_stats.update(averaged_stats)
        # --- END MODIFICATION ---

        # --- Get PZT Stats ---
        pzt_stats = get_pzt_data(pzt_filepath) # Returns a dict

        # Determine Load Value
        load_val = pzt_stats['load'] # Default to PZT file's load
        if load_col and load_col in row and not pd.isna(row[load_col]):
            log_load = pd.to_numeric(row[load_col], errors='coerce')
            load_val = log_load if not pd.isna(log_load) else load_val
        pzt_stats['load'] = load_val # Update the load in the dict

        # Only add if we have amplitude
        if not np.isnan(pzt_stats['mean_amplitude']):
            # Combine PZT stats and Strain stats into one record
            record = {'cycle': cycle, **pzt_stats, **final_strain_stats}
            aligned_records.append(record)
            processed_pzt_files.add(pzt_filename_log)


    if not aligned_records:
        print(f"  > Error: No valid records created using the LogBook."); return pd.DataFrame()

    df = pd.DataFrame(aligned_records)
    df = df.sort_values('cycle')

    # --- START MODIFICATION: Aggregate means correctly ---
    # Define aggregation functions for all columns
    agg_funcs = {
        'mean_amplitude': 'mean',
        'load': 'first', # Keep first load value for the cycle
        'pzt_amplitude_std': 'mean', # Average std dev if multiple reads
        'pzt_energy_mean': 'mean',   # Average energy if multiple reads
        'pzt_rms_mean': 'mean',      # Average RMS if multiple reads
        'median_strain': 'mean',     # Average median strain if multiple reads
        'strain_mean_of_means': 'mean', # Average mean strain if multiple reads
        'strain_std_of_means': 'mean'  # Average std dev if multiple reads
    }
    df = df.groupby('cycle', as_index=False).agg(agg_funcs)
    # --- END MODIFICATION ---

    # --- Final Cleanup ---
    # Define all columns to be filled and checked
    feature_cols = [
        'mean_amplitude', 'load', 'median_strain',
        'pzt_amplitude_std', 'pzt_energy_mean', 'pzt_rms_mean',
        'strain_mean_of_means', 'strain_std_of_means'
    ]
    # Fill NaNs robustly for all feature columns
    for col in feature_cols:
        if col in df.columns: # Check if column exists (e.g., std dev might be missing)
            df[col] = df[col].ffill().bfill()

    # Drop rows if *both* mean_amplitude and median_strain are NaN AFTER filling
    df = df.dropna(subset=['mean_amplitude', 'median_strain'], how='all')
    # Also drop if load is NaN
    df = df.dropna(subset=['load'])

    if df.empty:
        print(f"  > Error: No rows remaining after NaN handling for {specimen_id}.")

    print(f"  > Successfully loaded {len(df)} aligned data points.")
    # Return all potentially generated columns (some might be all NaN if data was missing)
    return df[['cycle'] + [col for col in feature_cols if col in df.columns]]