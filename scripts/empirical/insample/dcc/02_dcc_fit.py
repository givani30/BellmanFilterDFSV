import mgarch
import numpy as np
import joblib
import os
import pathlib

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "garch_outputs.npz")
OUTPUT_MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
OUTPUT_CORR_FILE = os.path.join(DATA_DIR, "Rt.npy")

# Load outputs from univariate GARCH
try:
    dat = np.load(INPUT_FILE)
    eps = dat['eps']
except Exception as e:
    print(f"Error loading univariate GARCH outputs: {e}")
    exit()

# Fit DCC-GARCH model
print("Fitting DCC-GARCH model...")
try:
    dcc = mgarch.mgarch(dist='t')  # Student-t innovations
    dcc.fit(eps)
    print("DCC-GARCH model fitted successfully.")
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

# Calculate and save the time series of conditional correlations
try:
    # Calculate the correlation matrix for each time point
    T = dcc.T
    N = dcc.N
    a = dcc.a
    b = dcc.b
    D_t = dcc.D_t

    # Standardized residuals
    eps_tilde = np.zeros((T, N))
    for i in range(T):
        for j in range(N):
            eps_tilde[i, j] = dcc.rt[i, j] / D_t[i, j]

    # Unconditional correlation matrix
    Q_bar = np.zeros((N, N))
    for i in range(T):
        Q_bar = Q_bar + np.outer(eps_tilde[i, :], eps_tilde[i, :])
    Q_bar = Q_bar / T

    # Initialize Q and R matrices
    Q = np.zeros((T, N, N))
    R = np.zeros((T, N, N))

    # First period uses unconditional correlation
    Q[0] = Q_bar
    Q_diag_inv = np.diag(1 / np.sqrt(np.diag(Q[0])))
    R[0] = Q_diag_inv @ Q[0] @ Q_diag_inv

    # Calculate dynamic conditional correlations
    for t in range(1, T):
        Q[t] = (1 - a - b) * Q_bar + a * np.outer(eps_tilde[t-1, :], eps_tilde[t-1, :]) + b * Q[t-1]
        Q_diag_inv = np.diag(1 / np.sqrt(np.diag(Q[t])))
        R[t] = Q_diag_inv @ Q[t] @ Q_diag_inv

    # Save the 3D correlation matrix (T, N, N) directly using numpy
    np.save(OUTPUT_CORR_FILE, R)
    print(f"Conditional correlations saved to {OUTPUT_CORR_FILE}")
except Exception as e:
    print(f"Error saving conditional correlations: {e}")
    print(f"Error details: {str(e)}")
    import traceback
    traceback.print_exc()
    exit()
