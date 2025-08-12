# Plan: Batch Optimization Study Framework

**Date:** 11-04-2025

**Goal:** Evaluate the performance of BIF and PF filters combined with AdamW and DampedTrustRegionBFGS optimizers in estimating DFSV model parameters (Λ, Φ_f, Φ_h, Q_h, Σ_ε) under various simulation settings (N, K, T), using Google Cloud Batch for execution. The script must support fixing or estimating `mu`.

## Plan Phases

### Phase 1: Script Creation & Preparation

1.  **Create New Script:** Create a *new* script named `scripts/run_optimization_replicate.py` (leaving `unified_filter_optimization.py` untouched for now).
2.  **Implement Argument Parsing:** Use `argparse` to accept configuration parameters:
    *   `--N`, `--K`, `--T`
    *   `--filter_type` (choices: `['BIF', 'PF']`)
    *   `--optimizer_name` (choices: `['AdamW', 'DampedTrustRegionBFGS']`, extensible)
    *   `--num_particles` (required if `filter_type=='PF'`)
    *   `--stability_penalty` (float, default: 1000.0)
    *   `--max_steps` (int, **default: 1000**)
    *   `--replicate_seed` (int)
    *   `--fix_mu` (bool, action='store_true')
    *   `--output_dir` (str)
    *   `--save_format` (str, default: 'pkl')
3.  **Implement `main` Function:** Structure the `main` function in the new script to use parsed arguments.
4.  **Ensure Parameter Stability:**
    *   Review and modify `create_simple_model` (used in the script) to **guarantee** stable `Phi_f` and `Phi_h` matrices (max eigenvalue magnitude strictly < 1, e.g., via normalization).
    *   Review and modify `create_initial_params` (used by `run_optimization`) to ensure the initial guess parameters provided to the optimizer are also stable.
5.  **Implement `mu` Handling:** Modify the optimization setup (likely within `run_optimization` or its caller) to conditionally fix `mu` based on the `--fix_mu` flag and true parameters, aligning with Decision [04-06-2025 17:40:11].
6.  **Calculate State Estimation Accuracy:** After successful optimization, run the filter with *estimated* params on the original data to get `f_hat`/`h_hat`. Use `calculate_accuracy` (from `run_config_batch.py`) to compute RMSE/Correlation against true states.
7.  **Refine Output Saving:**
    *   Save summary metrics (config, opt results, loss@true, param errors, state accuracy) to JSON per replicate (e.g., `metrics_N{N}_K{K}_{filter}_{optimizer}_rep{seed}.json`).
    *   Save `true_params` and `final_params` together in a separate file (e.g., `params_N{N}_K{K}_{filter}_{optimizer}_rep{seed}.{save_format}`).
8.  **Ensure GCS Compatibility:** Integrate `gcsfs` for writing outputs to GCS.
9.  **(NEW) Refactor Reusable Code:** Identify reusable functions (`create_simple_model`, `create_initial_params`, `calculate_accuracy`, potentially others) currently in scripts. Plan and implement their migration into the main codebase (e.g., `src/bellman_filter_dfsv/models/simulation_helpers.py` or `src/bellman_filter_dfsv/utils/analysis.py`) to promote reuse and simplify scripts. Rename functions for clarity if needed (e.g., `create_stable_dfsv_params`).

### Phase 2: Configuration & Batch Infrastructure

1.  Create `scripts/generate_optimization_configs.py` to define the parameter grid and generate replicate configurations for `run_optimization_replicate.py`.
2.  Review/update `Dockerfile`, build, and push to Artifact Registry.
3.  Create Cloud Batch job template (JSON/YAML) specifying tasks, hardware, container, args for the *new* script.
4.  Create `scripts/submit_optimization_batch_job.py` using `google-cloud-batch` library to generate tasks from configurations and submit the job.

### Phase 3: Execution & Analysis

1.  Launch job via submission script.
2.  Monitor job progress.
3.  Create `scripts/aggregate_optimization_results.py` to download JSON metrics from GCS and compile into a single CSV.
4.  Analyze aggregated results.

### Phase 4: Documentation

1.  Document the new workflow, configuration, submission, and analysis process.
2.  Document output file formats.

## Workflow Diagram

```mermaid
graph TD
    subgraph "Setup & Configuration"
        A[Define Parameter Space\n(generate_optimization_configs.py)] --> B(Generate Replicate Config List);
        C[Create Script\n(run_optimization_replicate.py)] --> D;
        E[Update Dockerfile] --> D[Build & Push Docker Image];
        F[Create Batch Job Template] --> G;
        B --> G[Create Batch Submission Script\n(submit_optimization_batch_job.py)];
    end

    subgraph "Execution (Google Cloud)"
        G -- Submits Job --> H{Cloud Batch Service};
        H -- Creates Tasks --> I(Batch Task Execution);
        I -- Runs --> D;
        I -- Writes Results --> J[Individual Results on GCS\n(JSON Metrics, PKL/NPZ Params)];
    end

    subgraph "Analysis"
        K[Aggregate Results Script\n(aggregate_results.py)] -- Reads --> J;
        K --> L[Aggregated Results CSV];
        L --> M[Analysis & Visualization];
    end

    subgraph "Documentation"
        N[Update Project Docs];
    end

    Setup & Configuration --> Execution (Google Cloud);
    Execution (Google Cloud) --> Analysis;
    Analysis --> Documentation;
```

## Batch Task Granularity

One Cloud Batch task per replicate configuration for maximum parallelism and fault isolation.