# 1_load_data.py
import scipy.io as sio
import pandas as pd
import numpy as np
from pathlib import Path

def load_pzt_file(filepath):
    """Extract relevant info from a single PZT .mat file."""
    mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    coupon = mat['coupon']
    cycle = int(coupon.cycles)
    condition_map = {"Baseline": 0, "Loaded": 1, " Clamped": 2," Traction-free": 3}
    cond_str = coupon.condition.strip()
    cond = condition_map.get(cond_str, -1)
    path_data = coupon.path_data

    amplitudes, freqs, sensors, actuators = [], [], [], []
    for p in np.atleast_1d(path_data):
        amplitudes.append(p.amplitude)
        freqs.append(p.frequency)
        sensors.append(p.sensor)
        actuators.append(p.actuator)
    
    # Flatten features to scalar summaries
    mean_amp = np.mean(amplitudes)
    std_amp = np.std(amplitudes)
    
    return {
        "cycle": cycle,
        "condition": cond,
        "mean_amplitude": mean_amp,
        "std_amplitude": std_amp
    }

def load_strain_file(filepath):
    """Extract robust (median) strain reading for each file."""
    mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    keys = [k for k in mat.keys() if k.startswith("strain")]
    med = np.median([np.median(mat[k]) for k in keys if np.size(mat[k]) > 0])
    return {"median_strain": med}


def aggregate_coupon_data(pzt_folder, strain_folder):
    """Merge PZT & strain data by cycle for one specimen."""
    pzt_records, strain_records = [], []

    for f in Path(pzt_folder).glob("*.mat"):
        if "_STRAIN" not in f.name:
            pzt_records.append(load_pzt_file(f))

    for f in Path(strain_folder).glob("*.mat"):
        strain_records.append(load_strain_file(f))
    
    df_pzt = pd.DataFrame(pzt_records).sort_values("cycle")
    df_strain = pd.DataFrame(strain_records)
    df = pd.concat([df_pzt.reset_index(drop=True), df_strain], axis=1)
    return df
