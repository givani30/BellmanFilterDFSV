import argparse
import json
from pathlib import Path
import numpy as np
import sys

def compare_metrics(metrics1, metrics2, tolerance=1e-6):
    """Compares two metrics dictionaries with numerical tolerance."""
    if metrics1.keys() != metrics2.keys():
        print("  - Metrics keys mismatch!")
        print(f"    Keys 1: {sorted(metrics1.keys())}")
        print(f"    Keys 2: {sorted(metrics2.keys())}")
        return False

    match = True
    for key in metrics1:
        # --- Skip timing keys ---
        if key in ['bf_time', 'pf_time']:
            continue
        # --- End skip ---
        val1 = metrics1[key]
        val2 = metrics2[key]

        # Handle lists/arrays (like RMSE/Corr)
        if isinstance(val1, list) and isinstance(val2, list):
            val1 = np.array(val1)
            val2 = np.array(val2)

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            # Replace None/NaN with a placeholder for comparison if needed, or check shape first
            if val1.shape != val2.shape:
                 print(f"  - Metrics['{key}'] shape mismatch: {val1.shape} vs {val2.shape}")
                 match = False
                 continue
            # Handle potential NaNs before comparison
            nan_mask1 = np.isnan(val1)
            nan_mask2 = np.isnan(val2)
            if not np.array_equal(nan_mask1, nan_mask2):
                 print(f"  - Metrics['{key}'] NaN pattern mismatch.")
                 match = False
                 continue
            # Compare non-NaN values
            if not np.allclose(val1[~nan_mask1], val2[~nan_mask2], rtol=tolerance, atol=tolerance, equal_nan=True):
                print(f"  - Metrics['{key}'] numerical mismatch (Array):")
                print(f"    Val 1: {val1}")
                print(f"    Val 2: {val2}")
                match = False
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if not np.allclose(val1, val2, rtol=tolerance, atol=tolerance, equal_nan=True):
                print(f"  - Metrics['{key}'] numerical mismatch:")
                print(f"    Val 1: {val1}")
                print(f"    Val 2: {val2}")
                match = False
        elif val1 != val2:
             # Handle case where one is None and the other isn't, or other type mismatches
             if (val1 is None) != (val2 is None): # Check if None mismatch
                 print(f"  - Metrics['{key}'] mismatch (None vs Value):")
                 print(f"    Val 1: {val1}")
                 print(f"    Val 2: {val2}")
                 match = False
             elif type(val1) != type(val2):
                 print(f"  - Metrics['{key}'] type mismatch: {type(val1)} vs {type(val2)}")
                 print(f"    Val 1: {val1}")
                 print(f"    Val 2: {val2}")
                 match = False
             elif key != 'error': # Don't strictly compare error messages if they exist
                 print(f"  - Metrics['{key}'] mismatch:")
                 print(f"    Val 1: {val1}")
                 print(f"    Val 2: {val2}")
                 match = False
    return match

def compare_npz_files(path1, path2, tolerance=1e-6):
    """Compares two NPZ files."""
    try:
        with np.load(path1) as data1, np.load(path2) as data2:
            if set(data1.files) != set(data2.files):
                print(f"  - NPZ keys mismatch!")
                print(f"    Keys 1: {sorted(data1.files)}")
                print(f"    Keys 2: {sorted(data2.files)}")
                return False

            match = True
            for key in data1.files:
                arr1 = data1[key]
                arr2 = data2[key]
                if arr1.shape != arr2.shape:
                    print(f"  - NPZ['{key}'] shape mismatch: {arr1.shape} vs {arr2.shape}")
                    match = False
                elif not np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance, equal_nan=True):
                    print(f"  - NPZ['{key}'] numerical mismatch.")
                    # Avoid printing large arrays, maybe print diff summary?
                    # print(f"    Arr 1: {arr1}")
                    # print(f"    Arr 2: {arr2}")
                    diff = np.abs(arr1 - arr2)
                    print(f"    Max Abs Diff: {np.nanmax(diff) if np.isnan(diff).any() else np.max(diff)}")
                    match = False
            return match
    except FileNotFoundError:
        print(f"  - NPZ file not found in one of the directories ({path1.name} or {path2.name})")
        return False
    except Exception as e:
        print(f"  - Error comparing NPZ files {path1.name} and {path2.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compare results from two simulation study directories.')
    parser.add_argument('dir1', type=str, help='Path to the first study directory.')
    parser.add_argument('dir2', type=str, help='Path to the second study directory.')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Relative and absolute tolerance for numerical comparisons.')
    args = parser.parse_args()

    path1 = Path(args.dir1)
    path2 = Path(args.dir2)
    tolerance = args.tolerance

    if not path1.is_dir() or not path2.is_dir():
        print("Error: One or both provided paths are not valid directories.")
        sys.exit(1)

    print(f"Comparing study directories:")
    print(f"1: {path1}")
    print(f"2: {path2}")
    print(f"Tolerance: {tolerance}\n")

    # 1. Compare simulation_config.json (excluding timestamp in dir name if stored)
    config1_path = path1 / "simulation_config.json"
    config2_path = path2 / "simulation_config.json"
    config_match = True
    try:
        with open(config1_path, 'r') as f1, open(config2_path, 'r') as f2:
            cfg1 = json.load(f1)
            cfg2 = json.load(f2)
            # Optionally ignore base_results_dir if it contains the study timestamp
            # cfg1.pop('base_results_dir', None)
            # cfg2.pop('base_results_dir', None)
            if cfg1 != cfg2:
                print("❌ Simulation config files differ.")
                # Could add detailed diff here if needed
                config_match = False
            else:
                print("✅ Simulation config files match.")
    except Exception as e:
        print(f"❌ Error comparing simulation config files: {e}")
        config_match = False

    # 2. Compare replicate subdirectories
    subdirs1 = sorted([d.name for d in path1.iterdir() if d.is_dir() and d.name.startswith('config_')])
    subdirs2 = sorted([d.name for d in path2.iterdir() if d.is_dir() and d.name.startswith('config_')])

    if subdirs1 != subdirs2:
        print("\n❌ Replicate subdirectory lists differ:")
        print(f"  Dirs in {path1.name}: {subdirs1}")
        print(f"  Dirs in {path2.name}: {subdirs2}")
        sys.exit(1) # Stop comparison if directory structure differs fundamentally

    print(f"\nFound {len(subdirs1)} matching replicate subdirectories. Comparing contents...")

    all_match = config_match # Start with config match status
    for subdir_name in subdirs1:
        print(f"\nComparing: {subdir_name}")
        rep_path1 = path1 / subdir_name
        rep_path2 = path2 / subdir_name

        metrics_match = False
        npz_match = False

        # Compare metrics.json
        metrics1_path = rep_path1 / "metrics.json"
        metrics2_path = rep_path2 / "metrics.json"
        error_metrics1_path = rep_path1 / "metrics_error.json"
        error_metrics2_path = rep_path2 / "metrics_error.json"

        if metrics1_path.exists() and metrics2_path.exists():
            try:
                with open(metrics1_path, 'r') as f1, open(metrics2_path, 'r') as f2:
                    m1 = json.load(f1)
                    m2 = json.load(f2)
                    if compare_metrics(m1, m2, tolerance):
                        print("  ✅ metrics.json files match.")
                        metrics_match = True
                    else:
                        print("  ❌ metrics.json files differ.")
                        all_match = False
            except Exception as e:
                print(f"  ❌ Error comparing metrics.json: {e}")
                all_match = False
        elif error_metrics1_path.exists() and error_metrics2_path.exists():
             print("  ✅ metrics_error.json found in both (skipped content check).")
             metrics_match = True # Assume error states are equivalent for this test
        elif metrics1_path.exists() != metrics2_path.exists():
             print(f"  ❌ metrics.json existence mismatch (Dir1: {metrics1_path.exists()}, Dir2: {metrics2_path.exists()})")
             all_match = False
        elif error_metrics1_path.exists() != error_metrics2_path.exists():
             print(f"  ❌ metrics_error.json existence mismatch (Dir1: {error_metrics1_path.exists()}, Dir2: {error_metrics2_path.exists()})")
             all_match = False
        else:
             print(f"  ❓ Neither metrics.json nor metrics_error.json found in {subdir_name} (might be ok if skipped).")
             metrics_match = True # Treat as match if neither exists

        # Compare raw_data.npz (only if metrics matched or were errors)
        if metrics_match:
            npz1_path = rep_path1 / "raw_data.npz"
            npz2_path = rep_path2 / "raw_data.npz"

            # Check existence first
            npz1_exists = npz1_path.exists()
            npz2_exists = npz2_path.exists()

            if npz1_exists and npz2_exists:
                if compare_npz_files(npz1_path, npz2_path, tolerance):
                    print("  ✅ raw_data.npz files match.")
                    npz_match = True
                else:
                    print("  ❌ raw_data.npz files differ.")
                    all_match = False
            elif npz1_exists != npz2_exists:
                 # Only flag as error if metrics.json existed (i.e., not an error run)
                 if metrics1_path.exists() and metrics2_path.exists():
                     print(f"  ❌ raw_data.npz existence mismatch (Dir1: {npz1_exists}, Dir2: {npz2_exists})")
                     all_match = False
                 else:
                     print(f"  ✅ raw_data.npz existence mismatch (likely due to error run, skipped).")
                     npz_match = True # Match if due to error
            else:
                 # Neither exists - this is fine if it was an error run
                 if metrics1_path.exists() and metrics2_path.exists():
                      print(f"  ❓ raw_data.npz not found in either directory for {subdir_name} (was data expected?).")
                      # Decide if this is an error? For now, assume ok.
                      npz_match = True
                 else:
                      print(f"  ✅ raw_data.npz not found in either directory (likely due to error run, skipped).")
                      npz_match = True # Match if due to error

    print("\n--- Comparison Summary ---")
    if all_match:
        print("✅✅✅ All compared files match within tolerance! ✅✅✅")
    else:
        print("❌❌❌ Differences found between study directories! ❌❌❌")
        sys.exit(1) # Exit with error code if differences found

if __name__ == "__main__":
    main()