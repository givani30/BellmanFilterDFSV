import polars as pl
import pandas as pd
from arch import arch_model
from joblib import Parallel, delayed
import numpy as np
import os
import pathlib

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
# Define paths relative to the script location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
INPUT_FILE = os.path.join(SCRIPT_DIR.parent.parent, "vw_returns_final_with_date.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "garch_outputs.npz")
DATE_INDEX_FILE = os.path.join(DATA_DIR, "date_index.txt") # To save date index

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load data using polars
# Assuming the first column is the date and all others are returns
try:
    df_pl = pl.read_csv(INPUT_FILE)
    date_col_name = df_pl.columns[0]
    return_cols = df_pl.columns[1:]

    # Convert to pandas DataFrame for arch, setting date as index
    df_pd = df_pl.with_columns(pl.col(date_col_name).str.strptime(pl.Date, "%Y-%m-%d").cast(pl.Datetime)).to_pandas()
    df_pd = df_pd.set_index(date_col_name)
    R = df_pd[return_cols]

except Exception as e:
    print(f"Error loading or processing data: {e}")
    exit()

# Save the date index
try:
    with open(DATE_INDEX_FILE, 'w') as f:
        for date in R.index:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")
except Exception as e:
    print(f"Error saving date index: {e}")
    exit()


def fit_one(x):
    """Fits a univariate GARCH(1,1)-t model to a single series."""
    # Ensure input is a pandas Series
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    try:
        res = arch_model(x, vol='Garch', p=1, q=1, dist='t').fit(disp='off')
        return pd.DataFrame({'sigma2': res.conditional_volatility**2,
                             'resid' : res.resid / res.conditional_volatility})
    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
        # Return empty DataFrame or handle error as appropriate
        return pd.DataFrame({'sigma2': np.full(len(x), np.nan),
                             'resid' : np.full(len(x), np.nan)})


# Parallel univariate GARCH(1,1) fitting
print("Fitting univariate GARCH models...")
out = Parallel(n_jobs=-1)(delayed(fit_one)(R[col]) for col in R.columns)

# Concatenate results and save
sigma2 = pd.concat([o.sigma2 for o in out], axis=1).to_numpy()
eps    = pd.concat([o.resid  for o in out], axis=1).to_numpy()

try:
    np.savez(OUTPUT_FILE, sigma2=sigma2, eps=eps)
    print(f"Univariate GARCH outputs saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"Error saving GARCH outputs: {e}")
