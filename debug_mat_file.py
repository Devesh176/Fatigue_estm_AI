import scipy.io as sio
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
# Put one of the .mat files you uploaded into the same directory
# as this script.
FILE_TO_INSPECT = 'L3S18_1000_1_2.mat' 
# (or 'L3S18_0_1_9.mat')
# ---

def print_struct(obj, indent=0):
    """Recursively prints the structure of a loaded .mat file."""
    prefix = '  ' * indent

    # Case 1: Object is a dict (top level)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.startswith('__'): continue # Skip metadata
            print(f"{prefix}[DICT KEY] {key}:")
            print_struct(value, indent + 1)

    # Case 2: Object is a MATLAB struct (loaded as an object)
    elif hasattr(obj, '__dict__'):
        for field in obj.__dict__:
            if field.startswith('__'): continue
            print(f"{prefix}[STRUCT FIELD] {field}:")
            print_struct(getattr(obj, field), indent + 1)

    # Case 3: Object is a numpy array
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}Numpy Array | Shape: {obj.shape} | Dtype: {obj.dtype}")
        # If it's an array of objects, print the first one
        if obj.dtype == 'object' and obj.size > 0:
            print(f"{prefix}  > Printing structure of first element (as example):")
            print_struct(obj.flat[0], indent + 2)

    # Base Case: All other types
    else:
        # Print the value, but truncate if it's too long
        value_str = str(obj)
        if len(value_str) > 100:
            value_str = value_str[:100] + "..."
        print(f"{prefix}Value: {value_str} | Type: {type(obj).__name__}")

# --- Main execution ---
print(f"--- Inspecting structure of: {FILE_TO_INSPECT} ---")

file_path = Path(FILE_TO_INSPECT)

if not file_path.exists():
    print(f"Error: File not found: {FILE_TO_INSPECT}")
    print("Please make sure the file is in the same directory as this script.")
else:
    try:
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        print_struct(mat)
    except Exception as e:
        print(f"Error: Could not read {FILE_TO_INSPECT}. {e}")

print("\n--- End of Inspection ---")