#!/usr/bin/env python
"""
Fix simulation results by properly parsing array strings without commas.
This script reads the original simulation_results.csv, fixes the array parsing,
and saves the processed data.
"""

import pandas as pd
import numpy as np
from simulation_analysis import parse_array_string, process_array_columns

def main():
    print("Reading simulation results file...")
    results_df = pd.read_csv('simulation_results.csv')
    
    print("Initial data shape:", results_df.shape)
    
    # Define array columns
    array_columns = ['bf_rmse_f', 'bf_corr_f', 'bf_rmse_h', 'bf_corr_h',
                    'pf_rmse_f', 'pf_corr_f', 'pf_rmse_h', 'pf_corr_h']
    
    # Process the array columns
    print("Processing array columns...")
    results_df = process_array_columns(results_df, array_columns)
    
    # Separate BF and PF results for aggregation
    bf_results = results_df[results_df['num_particles'].isna()].copy()
    pf_results = results_df[~results_df['num_particles'].isna()].copy()
    
    print(f"Bellman filter rows: {len(bf_results)}")
    print(f"Particle filter rows: {len(pf_results)}")
    
    # Aggregate Bellman filter results
    bf_agg = bf_results.groupby(['N', 'K']).agg({
        'bf_time': 'mean',
        'bf_corr_f_mean': 'mean',
        'bf_corr_h_mean': 'mean',
        'bf_rmse_f_mean': 'mean',
        'bf_rmse_h_mean': 'mean'
    }).reset_index()
    
    # Aggregate Particle filter results
    pf_agg = pf_results.groupby(['N', 'K', 'num_particles']).agg({
        'pf_time': 'mean',
        'pf_corr_f_mean': 'mean',
        'pf_corr_h_mean': 'mean',
        'pf_rmse_f_mean': 'mean',
        'pf_rmse_h_mean': 'mean'
    }).reset_index()
    
    # Save processed data
    results_df.to_csv('processed_simulation_results.csv', index=False)
    bf_agg.to_csv('bellman_filter_results.csv', index=False)
    pf_agg.to_csv('particle_filter_results.csv', index=False)
    
    print("Saved processed data to:")
    print("- processed_simulation_results.csv (all data)")
    print("- bellman_filter_results.csv (aggregated BF results)")
    print("- particle_filter_results.csv (aggregated PF results)")

if __name__ == "__main__":
    main()