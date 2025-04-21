# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Empirical Data Exploratory Analysis
#
# This notebook contains exploratory data analysis for empirical data.

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import jax.numpy as jnp
from pathlib import Path

# Enable better plotting
sns.set_theme(style="darkgrid")

# %%
# Add your data loading code here
