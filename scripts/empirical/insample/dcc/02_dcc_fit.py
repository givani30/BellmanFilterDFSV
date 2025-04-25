import mgarch
import numpy as np
import pandas as pd
import joblib
import os
import pathlib
import time
import json

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RETURNS_FILE = os.path.join(SCRIPT_DIR.parent.parent, "vw_returns_final_with_date.csv")
OUTPUT_MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
OUTPUT_CORR_FILE = os.path.join(DATA_DIR, "Rt.npy")
OUTPUT_COV_FILE = os.path.join(DATA_DIR, "Ht.npy")
OUTPUT_METADATA_FILE = os.path.join(DATA_DIR, "model_metadata.json")
DATE_INDEX_FILE = os.path.join(DATA_DIR, "date_index.txt")

# Load original returns data
try:
    # Load returns data
    df = pd.read_csv(RETURNS_FILE)
    date_col = df.columns[0]

    # Convert date column to datetime and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    # Extract returns (all columns except date)
    returns = df.values

    # Save date index for later use
    with open(DATE_INDEX_FILE, 'w') as f:
        for date in df.index:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")

    # Apply scaling for better numerical stability (as recommended by arch package)
    # This is consistent with the scaling in 01_univariate_garch.py
    returns = returns * 10.0

    print(f"Loaded returns data with shape {returns.shape}")
except Exception as e:
    print(f"Error loading returns data: {e}")
    exit()

# Fit DCC-GARCH model
print("Fitting DCC-GARCH model...")
try:
    # Track estimation time
    start_time = time.time()

    # Initialize model
    dcc = mgarch.mgarch(dist='t')  # Student-t innovations

    # Fit model with original returns
    result = dcc.fit(returns)

    # Calculate estimation time
    estimation_time = time.time() - start_time

    # Determine convergence status
    if hasattr(dcc, 'a') and hasattr(dcc, 'b') and hasattr(dcc, 'dof'):
        convergence_status = "Converged"
    else:
        convergence_status = "Failed"

    print(f"DCC-GARCH model fitted successfully in {estimation_time:.2f} seconds.")
    print(f"Convergence status: {convergence_status}")

    # Create metadata dictionary
    T, N = returns.shape
    metadata = {
        "model_type": "DCC-GARCH",
        "distribution": "Student-t",
        "num_params": N * 3 + 3,  # 3 per univariate GARCH + (a, b, nu)
        "estimation_time": estimation_time,
        "convergence_status": convergence_status,
        "sample_size": T,
        "num_series": N,
        "parameters": {
            "a": float(dcc.a),
            "b": float(dcc.b),
            "dof": float(dcc.dof)
        }
    }

except Exception as e:
    print(f"Error fitting DCC-GARCH model: {e}")
    exit()

# Save the fitted model
try:
    joblib.dump(dcc, OUTPUT_MODEL_FILE)
    print(f"Fitted DCC-GARCH model saved to {OUTPUT_MODEL_FILE}")
except Exception as e:
    print(f"Error saving fitted model: {e}")
    exit()

# Extract and save the conditional correlations and covariances from the fitted model
try:
    # Get dimensions
    T, N = returns.shape

    # Extract standardized residuals from the model
    # These are the residuals after the model's internal GARCH fitting
    eps_tilde = np.zeros((T, N))
    for i in range(T):
        for j in range(N):
            # dcc.rt contains the original returns
            # dcc.D_t contains the conditional standard deviations from the model's internal GARCH fits
            eps_tilde[i, j] = dcc.rt[i, j] / dcc.D_t[i, j]

    # Extract conditional correlations (R) and calculate conditional covariances (H)
    # The mgarch model doesn't directly store R and H as attributes, so we need to calculate them

    # Initialize R and H matrices
    R = np.zeros((T, N, N))
    H = np.zeros((T, N, N))

    # Get DCC parameters
    a = dcc.a
    b = dcc.b

    # Calculate unconditional correlation matrix (Q_bar)
    Q_bar = np.zeros((N, N))
    for i in range(T):
        Q_bar = Q_bar + np.outer(eps_tilde[i, :], eps_tilde[i, :])
    Q_bar = Q_bar / T

    # Initialize Q matrix for the recursion
    Q = np.zeros((T, N, N))

    # First period uses unconditional correlation
    Q[0] = Q_bar.copy()

    # Calculate R[0] from Q[0]
    Q_diag_inv = np.diag(1 / np.sqrt(np.diag(Q[0])))
    R[0] = Q_diag_inv @ Q[0] @ Q_diag_inv

    # Ensure diagonal elements are exactly 1
    np.fill_diagonal(R[0], 1.0)

    # Calculate H[0]
    D_t_diag = np.diag(dcc.D_t[0])
    H[0] = D_t_diag @ R[0] @ D_t_diag

    # Calculate dynamic conditional correlations and covariances
    for t in range(1, T):
        # The DCC recursion for Q_t
        Q[t] = (1 - a - b) * Q_bar + a * np.outer(eps_tilde[t-1, :], eps_tilde[t-1, :]) + b * Q[t-1]

        # Calculate R[t] from Q[t]
        Q_diag_inv = np.diag(1 / np.sqrt(np.diag(Q[t])))
        R[t] = Q_diag_inv @ Q[t] @ Q_diag_inv

        # Ensure R[t] is a valid correlation matrix
        np.fill_diagonal(R[t], 1.0)

        # Calculate H[t]
        D_t_diag = np.diag(dcc.D_t[t])
        H[t] = D_t_diag @ R[t] @ D_t_diag

    # Save the standardized residuals for diagnostics
    np.save(os.path.join(DATA_DIR, "eps_tilde.npy"), eps_tilde)
    print(f"Standardized residuals saved to {os.path.join(DATA_DIR, 'eps_tilde.npy')}")

    # Save the 3D correlation matrix (T, N, N)
    np.save(OUTPUT_CORR_FILE, R)
    print(f"Conditional correlations saved to {OUTPUT_CORR_FILE}")

    # Save the 3D covariance matrix (T, N, N)
    np.save(OUTPUT_COV_FILE, H)
    print(f"Conditional covariances saved to {OUTPUT_COV_FILE}")

    # Save metadata
    with open(OUTPUT_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {OUTPUT_METADATA_FILE}")

except Exception as e:
    print(f"Error saving model outputs: {e}")
    print(f"Error details: {str(e)}")
    import traceback
    traceback.print_exc()
    exit()
