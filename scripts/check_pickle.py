import pickle
import sys

# Define the path to the pickle file
pickle_file_path = '/home/givanib/Documents/BellmanFilterDFSV/batch_outputs/BF_runs/task_00000_BIF_N5_K2_seed2923803421/replicate_results_params.pkl'

try:
    # Open the file in binary read mode
    with open(pickle_file_path, 'rb') as f:
        # Load the data from the pickle file
        data = pickle.load(f)

    # Print the loaded data
    print(f"Successfully loaded data from: {pickle_file_path}")
    print("Contents:")
    print(data)

except FileNotFoundError:
    print(f"Error: File not found at {pickle_file_path}", file=sys.stderr)
except pickle.UnpicklingError:
    print(f"Error: Could not unpickle file {pickle_file_path}. It might be corrupted or not a valid pickle file.", file=sys.stderr)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)

