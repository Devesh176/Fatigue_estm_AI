from pathlib import Path

# --- !! CHANGE THIS !! ---
# Let's inspect one of the folders that failed.
# Please pick one from your log, for example: "data/Layup1/Coupon_L1_S11_F"
coupon_folder_path = Path("data/Layup2/Coupon_L2_S17_F")
# ---

pzt_path = coupon_folder_path / "PZT-data"
strain_path = coupon_folder_path / "StrainData"

print(f"--- Debugging File Names for: {coupon_folder_path} ---")

# --- List PZT-data files ---
print(f"\nListing files in: {pzt_path}")
if pzt_path.exists():
    pzt_files = [f.name for f in pzt_path.glob("*.mat")]
    if pzt_files:
        # Show up to 20 files
        print("\n".join(sorted(pzt_files)[:20]))
        if len(pzt_files) > 20:
            print(f"...and {len(pzt_files) - 20} more.")
    else:
        print("No .mat files found in this directory.")
else:
    print("PZT-data directory NOT FOUND.")

# --- List StrainData files ---
print(f"\nListing files in: {strain_path}")
if strain_path.exists():
    strain_files = [f.name for f in strain_path.glob("*")]
    if strain_files:
        # Show all strain files (usually fewer)
        print("\n".join(sorted(strain_files)))
    else:
        print("No .mat files found in this directory.")
else:
    print("StrainData directory NOT FOUND.")

print("\n--- End of Debug ---")