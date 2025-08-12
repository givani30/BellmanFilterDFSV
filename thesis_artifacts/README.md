# Thesis Artifacts Archive

This directory contains all the thesis-specific materials that were moved during the codebase cleanup process to create a more presentable and reusable package.

## Contents

### Research Documents
- `*.pdf` - Thesis documents, proposals, and reference papers
- `memory-bank/` - Research notes, decision logs, and development patterns
- `notebooks/` - Jupyter notebooks used for exploration and analysis

### Analysis and Results
- `outputs/` - Generated results, plots, and analysis outputs
- `output_thesis/` - Thesis-specific analysis results
- `final_out/` - Final analysis outputs for thesis
- `scripts/` - Analysis scripts and simulation studies
- `simulation1_data/` - Simulation study data

### Experimental Code
- `06_scheduler_comparison.py` through `10_advanced_stability_techniques.py` - Advanced examples
- `test_lax_while_optimization.py` - Experimental optimization tests
- `Archive/` - Archived test files
- `simstudy_2/` - Simulation study 2 materials

### Infrastructure
- `batch_configs/` - Batch processing configurations
- `batch_outputs/` - Batch processing results
- `batch_job_*.json` - Batch job templates
- `test_batch_results/` - Batch test results
- `Dockerfile` - Container configuration
- `cloudbatchconfig.txt` - Cloud batch configuration

### Data and Figures
- `figure*.png` - Generated figures and plots
- `*.csv` - Data files and results
- `tree.txt` - Directory structure snapshot
- `projectfiles.txt` - Project file listing

### Legacy Code
- `Bellman filter replication code/` - Original replication code
- `thesis_code_submission/` - Code submission for thesis

## Purpose

These materials were archived to:
1. **Clean the main codebase** - Remove thesis-specific clutter
2. **Preserve research work** - Keep all valuable analysis and results
3. **Enable reusability** - Make the core package more accessible
4. **Maintain history** - Preserve the complete development journey

## Accessing Archived Materials

If you need to access any of these materials:
1. They are preserved in the `thesis-archive` git branch
2. All files remain in this directory for reference
3. The core algorithms and implementations are still available in the main package

## Core Package Location

The cleaned, reusable core package is now located in:
- `../src/bellman_filter_dfsv/` - Main package
- `../examples/` - Clean, documented examples
- `../tests/` - Core functionality tests
- `../README.md` - Updated documentation

This archive ensures that no research work is lost while making the codebase more professional and reusable.
