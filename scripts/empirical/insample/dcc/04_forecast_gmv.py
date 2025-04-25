import numpy as np
import pandas as pd
import joblib
import os
import pathlib

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
RETURNS_FILE = os.path.join(SCRIPT_DIR.parent.parent, "vw_returns_final_with_date.csv")
EPS_TILDE_FILE = os.path.join(DATA_DIR, "eps_tilde.npy")
RT_FILE = os.path.join(DATA_DIR, "Rt.npy")
COV_FILE = os.path.join(DATA_DIR, "Ht.npy")
OUTPUT_WEIGHTS_FILE = os.path.join(DATA_DIR, "w_tplus1.npy")

# Load fitted model, returns, standardized residuals, and correlation matrices
try:
    # Load the DCC model
    dcc = joblib.load(MODEL_FILE)

    # Load the original returns (without scaling)
    df = pd.read_csv(RETURNS_FILE)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    returns = df.values  # Using original decimal returns

    # Load the standardized residuals from the DCC model
    eps_tilde = np.load(EPS_TILDE_FILE)

    # Load the correlation matrices
    Rt = np.load(RT_FILE)

    # Load the covariance matrices
    Sigma_t = np.load(COV_FILE)
except Exception as e:
    print(f"Error loading data or model: {e}")
    exit()

T, N = returns.shape

# Helper function to forecast univariate GARCH conditional variance
def forecast_garch_variance(omega, alpha, beta, last_variance, last_squared_residual, steps=1):
    """
    Forecasts the univariate GARCH conditional variance for a given number of steps.

    Args:
        omega, alpha, beta: GARCH parameters
        last_variance: Last observed conditional variance
        last_squared_residual: Last observed squared residual
        steps (int): The number of steps to forecast.

    Returns:
        float: The forecasted conditional variance.
    """
    variance = last_variance

    for _ in range(steps):
        variance = omega + alpha * last_squared_residual + beta * variance

    return variance


# One-step-ahead forecast for the GMV test
print("Calculating one-step-ahead forecast for GMV...")
try:
    a = dcc.a  # DCC parameter a
    b = dcc.b  # DCC parameter b

    # Calculate unconditional correlation matrix (Q_bar)
    Q_bar = np.zeros((N, N))
    for i in range(T):
        Q_bar = Q_bar + np.outer(eps_tilde[i, :], eps_tilde[i, :])
    Q_bar = Q_bar / T

    # Get the last correlation matrix and standardized residuals
    Rt_last = Rt[-1]
    eps_tilde_last = eps_tilde[-1, :]

    # Initialize Q matrix for the recursion
    Q = np.zeros((T, N, N))
    Q[0] = Q_bar.copy()

    # Calculate Q matrices using the DCC recursion
    for t in range(1, T):
        Q[t] = (1 - a - b) * Q_bar + a * np.outer(eps_tilde[t-1, :], eps_tilde[t-1, :]) + b * Q[t-1]

    # Get the last Q matrix
    Q_last = Q[-1]

    # Forecast Qt+1 using the DCC recursion
    Qt1 = (1 - a - b) * Q_bar + a * np.outer(eps_tilde_last, eps_tilde_last) + b * Q_last

    # Ensure Qt1 is a valid correlation matrix
    Q_diag_inv = np.diag(1 / np.sqrt(np.diag(Qt1)))
    Rt1 = Q_diag_inv @ Qt1 @ Q_diag_inv

    # Ensure diagonal elements are exactly 1
    np.fill_diagonal(Rt1, 1.0)

    # Forecast univariate GARCH variances
    # For simplicity, we'll use a constant forecast (last observed variance)
    # In a real application, you would use the GARCH parameters to forecast
    D_t_last = dcc.D_t[-1]
    D_t_next = D_t_last  # Simple forecast - use last observed values

    # Construct the forecasted covariance matrix
    D_t_next_diag = np.diag(D_t_next)
    Sigma_next = D_t_next_diag @ Rt1 @ D_t_next_diag

    # Calculate GMV weights
    # Handle potential singularity issues by adding a small ridge
    try:
        w_gmv = np.linalg.solve(Sigma_next, np.ones(N))
    except np.linalg.LinAlgError:
        print("Covariance matrix is singular, adding a small ridge for GMV calculation.")
        ridge = 1e-6 * np.eye(N)
        w_gmv = np.linalg.solve(Sigma_next + ridge, np.ones(N))

    w_gmv /= w_gmv.sum()  # Normalize weights to sum to 1

    print("GMV weights calculated.")

except Exception as e:
    print(f"Error during GMV forecast and calculation: {e}")
    exit()


# Save GMV weights
try:
    np.save(OUTPUT_WEIGHTS_FILE, w_gmv)
    print(f"GMV weights saved to {OUTPUT_WEIGHTS_FILE}")
except Exception as e:
    print(f"Error saving GMV weights: {e}")
