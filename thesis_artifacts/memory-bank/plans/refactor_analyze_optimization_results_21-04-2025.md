# Refactoring Plan for analyze_optimization_results.py

**Date:** 21-04-2025

**Goal:** Improve modularity, readability, and consistency of `scripts/simstudy_2/analyze_optimization_results.py` by splitting it into multiple scripts within `scripts/simstudy_2/`, centralizing output saving, centralizing plot styling, and adding unit tests.

**Approach:**

1.  **Identify Logical Units:** Break down the script into logical units suitable for separate files within `scripts/simstudy_2/`.
    *   **Data Utilities:** Loading, validation, preparation (`load_and_validate_metrics`, `load_and_validate_params`, `filter_successful_runs`, `calculate_loss_diff`).
    *   **Error Metrics:** Parameter error calculations (`calculate_comprehensive_param_errors`, `calculate_param_errors_for_replicate`, `create_param_errors_df`, `calculate_matrix_element_errors`).
    *   **Scalar Metrics:** Scalar metric calculations and aggregation (`calculate_scalar_metrics`, `aggregate_scalar_metrics`).
    *   **Plotting Configuration:** Centralized settings for seaborn/matplotlib (style, palette, fonts, sizes).
    *   **Plotting Utilities:** Functions that *generate* plots but don't save them (`plot_scatter_comparison`, `plot_error_heatmaps`, `plot_error_boxplots`, `plot_k2_eigenvalue_distributions`, `create_time_scaling_plots`). These will import and apply settings from the plotting configuration module and return plot objects (e.g., matplotlib Figures).
    *   **Table Utilities:** Functions that *generate* summary tables as DataFrames (`generate_summary_tables`). These will be modified to return DataFrames.
    *   **Specific Analyses:** Functions performing specific comparisons or analyses (`analyze_identification_difficulty`, `perform_comparative_analysis`, `analyze_fix_mu_effect`).
    *   **I/O Utilities:** Centralized functions for saving outputs and creating directories (`create_output_directory` and a new saving function).
    *   **Main Orchestration Script:** The script that drives the analysis workflow.
2.  **Create New Script Files:** Create new Python files within `scripts/simstudy_2/` for these units. Example names:
    *   `analysis_data_utils.py`
    *   `analysis_error_metrics.py`
    *   `analysis_scalar_metrics.py`
    *   `analysis_plotting_config.py`  *(New)*
    *   `analysis_plotting_utils.py`
    *   `analysis_table_utils.py`
    *   `analysis_specific_utils.py`
    *   `analysis_io_utils.py`
    *   `analyze_optimization_results_main.py` (Rename the original or create new)
3.  **Centralize Plotting Style:**
    *   Create `analysis_plotting_config.py`.
    *   Define dictionaries or functions within it to return consistent `matplotlib.rcParams` updates, seaborn styles/palettes, font settings, etc.
    *   Example: A function `apply_publication_style()` that sets all necessary parameters.
4.  **Centralize Saving Logic:**
    *   Create `analysis_io_utils.py`.
    *   Move `create_output_directory` into it.
    *   Create `save_analysis_output(output_object, filename, output_dir)` to handle saving Figures (PNG/PDF) and DataFrames (CSV), constructing the full path.
5.  **Move and Refactor Functions:**
    *   Distribute functions from the original script into the appropriate new utility scripts.
    *   Modify plotting functions in `analysis_plotting_utils.py` to:
        *   Import and call the style function from `analysis_plotting_config.py` at the beginning.
        *   `return fig` instead of saving/closing.
    *   Modify table generation functions in `analysis_table_utils.py` to `return df`.
6.  **Refactor Orchestration Script (`analyze_optimization_results_main.py`):**
    *   Keep `parse_args` and `setup_logging`.
    *   The `main` function will:
        *   Import functions from all utility scripts.
        *   Parse arguments, set up logging.
        *   Call `create_output_directory` from `analysis_io_utils`.
        *   Call data loading/prep functions.
        *   Call error/metric calculation functions.
        *   Call specific analysis functions.
        *   Call plotting functions (which now apply consistent style internally) to get Figure objects.
        *   Call table functions to get DataFrame objects.
        *   Call `save_analysis_output` repeatedly to save generated Figures and DataFrames.
7.  **Add Unit Tests:** *(New Step)*
    *   Create a corresponding test file for each new utility module (e.g., `tests/simstudy_2/test_analysis_data_utils.py`).
    *   Write basic unit tests for key functions within each module, focusing on:
        *   Correct data loading/parsing (mocking file reads if necessary).
        *   Correct calculation logic (using small, controlled inputs).
        *   Correct output types (e.g., Figure objects from plotting utils, DataFrames from table utils).
    *   These tests will primarily ensure the functions run without errors and handle basic cases, not exhaustive validation of the analysis itself.
8.  **Update Imports and Documentation:** Adjust imports and docstrings across all affected files (original script, new utility scripts, new test scripts).

**Revised Visual Representation (Mermaid):**

```mermaid
graph TD
    A[Original Script: analyze_optimization_results.py] --> B{Refactoring};

    subgraph New Utility Scripts in scripts/simstudy_2/
        C[analysis_data_utils.py];
        E[analysis_error_metrics.py];
        F[analysis_scalar_metrics.py];
        K[analysis_plotting_config.py];
        G[analysis_plotting_utils.py];
        H[analysis_table_utils.py];
        I[analysis_specific_utils.py];
        J[analysis_io_utils.py];
    end

    subgraph New Test Scripts in tests/simstudy_2/
        TC[test_analysis_data_utils.py];
        TE[test_analysis_error_metrics.py];
        TF[test_analysis_scalar_metrics.py];
        TG[test_analysis_plotting_utils.py];
        TH[test_analysis_table_utils.py];
        TI[test_analysis_specific_utils.py];
        TJ[test_analysis_io_utils.py];
    end

    subgraph Main Orchestration Script
        D[analyze_optimization_results_main.py];
        D1[main()];
        D2[parse_args()];
        D3[setup_logging()];
        D4{Imports & Calls Utility Funcs};
        D5{Receives Fig/DF Objects};
        D6{Calls Centralized Saving Util};
    end

    B --> C & E & F & G & H & I & J & K & D;
    B --> TC & TE & TF & TG & TH & TI & TJ;

    C --> C1[Data Loading/Prep Funcs];
    E --> E1[Error Calc Funcs];
    F --> F1[Scalar Metric Funcs];
    K --> K1[Plot Style Settings];
    G -- Imports --> K1;
    G --> G1[Plotting Funcs (Apply Style, Return Fig)];
    H --> H1[Table Gen Funcs (Return DF)];
    I --> I1[Specific Analysis Funcs];
    J --> J1[save_analysis_output()];
    J --> J2[create_output_directory()];

    TC -- Tests --> C1;
    TE -- Tests --> E1;
    TF -- Tests --> F1;
    TG -- Tests --> G1;
    TH -- Tests --> H1;
    TI -- Tests --> I1;
    TJ -- Tests --> J1 & J2;

    D --> D1 & D2 & D3 & D4 & D5 & D6;
    D4 --> C1 & E1 & F1 & G1 & H1 & I1 & J2;
    D5 --> G1 & H1;
    D6 --> J1;


    style C,E,F,G,H,I,J,K fill:#f9f,stroke:#333,stroke-width:2px;
    style TC,TE,TF,TG,TH,TI,TJ fill:#dfd,stroke:#333,stroke-width:2px;
    style D fill:#ccf,stroke:#333,stroke-width:2px;