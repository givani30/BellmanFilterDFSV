# Plan: Create Thesis Code Submission Folder (30-04-2025)

**Objective:** Create a clean, self-contained subfolder (`thesis_code_submission/`) containing the necessary code, processed data, and key outputs to enable reproducibility of the empirical analysis results and figures presented in the thesis, and verify its functionality.

**Reproducibility Scope:**
*   Users should be able to set up the environment within the `thesis_code_submission/` folder and run specific scripts to load processed data and saved model outputs (estimated parameters, filtered states, etc.).
*   Users should be able to regenerate the figures and tables presented in the empirical chapters of the thesis using these loaded data and outputs.
*   Re-running the full model estimation procedures (especially for DFSV-BIF/PF) and the extensive simulation studies is *not* expected due to computational cost and time constraints. The scripts for these will *not* be included in the submission folder to keep it focused and smaller.

**Plan Steps:**

1.  **Initial Assessment:** Briefly review the current directory structure to confirm the location of core code (`src/`), scripts (`scripts/`), data (`Data/`, `data/`), and outputs (`outputs/`, `Figures/`).
2.  **Create Submission Directory:** Create a new directory at the top level of the current working directory, named `thesis_code_submission/`.
3.  **Copy Core Code:** Copy the `src/bellman_filter_dfsv/` directory and its contents into `thesis_code_submission/src/`.
4.  **Identify and Copy Processed Data:** Locate the specific processed data file(s) used in the empirical analysis (e.g., `Data/Fama_french/100_portfolios_monthly/VW_Returns_Monthly.csv` or similar). Copy these files into `thesis_code_submission/Data/processed/`.
5.  **Identify and Copy Key Empirical Outputs:** Locate the saved results from the empirical model fitting (e.g., `.pkl` files containing estimated parameters and filtered states, `.csv` files from residual analysis within `outputs/empirical/`). Copy these files into `thesis_code_submission/outputs/empirical/`.
6.  **Identify and Copy Essential Empirical Scripts:** Review the `scripts/empirical/` directory. Identify the specific scripts needed to load the data/outputs from steps 4 and 5 and regenerate the empirical figures and tables. Copy these essential scripts into `thesis_code_submission/scripts/empirical/`. *Do not* copy scripts for full model estimation or simulation studies. Include local script dependencies like `multivariate_portmanteau.py`.
7.  **Create README.md (in submission folder):** Write a new `README.md` file within `thesis_code_submission/`.
    *   Provide a brief description of the folder's contents and purpose (thesis code submission for empirical reproducibility).
    *   Include clear instructions on how to set up the required Python environment (referencing the `requirements.txt` to be created).
    *   Explicitly state the level of reproducibility supported (empirical analysis from provided data/outputs).
    *   Provide clear, step-by-step instructions on which script(s) to run within `thesis_code_submission/scripts/empirical/` to regenerate the empirical results and figures.
    *   Include a link to the full GitHub repository for the project.
    *   Mention that the full thesis PDF (`G_Boekestijn_thesisdraft_final.pdf`) provides complete details on methodology, results, and interpretation.
8.  **Create requirements.txt (in submission folder):** Create a new `requirements.txt` file within `thesis_code_submission/`. List only the Python packages strictly necessary to run the scripts copied in step 6.
9.  **Test Submission Folder:**
    *   Navigate into the `thesis_code_submission/` directory.
    *   Set up the Python environment using the provided `requirements.txt` (using `uv venv` and activating the environment).
    *   Run the essential empirical scripts identified in step 6.
    *   Verify that the scripts execute without errors and successfully load the data and outputs from their respective paths within the `thesis_code_submission/` structure.
    *   Confirm that the scripts generate the expected output files (figures, tables).
10. **Final Submission Folder Review:** Perform a quick check of the `thesis_code_submission/` directory structure (`src/`, `scripts/empirical/`, `Data/processed/`, `outputs/empirical/`, `README.md`, `requirements.txt`) to ensure it is self-contained, logical, and contains all necessary files for reproducibility level 'b'.

```mermaid
graph TD
    A[Current Repository] --> B{Identify Necessary Files};
    B --> C[src/bellman_filter_dfsv/];
    B --> D[scripts/empirical/ (Essential Scripts)];
    B --> E[Data/ (Processed Data)];
    B --> F[outputs/empirical/ (Key Outputs)];

    subgraph Create Submission Folder
        G[thesis_code_submission/];
        G --> H[src/];
        G --> I[scripts/empirical/];
        G --> J[Data/processed/];
        G --> K[outputs/empirical/];
        G --> L[README.md];
        G --> M[requirements.txt];
    end

    C --> H[Copy src/];
    D --> I[Copy Essential Scripts];
    E --> J[Copy Processed Data];
    F --> K[Copy Key Outputs];

    L[Create README.md];
    M[Create requirements.txt];

    G --> N[Test Submission Folder];
    N --> O[Final Review];
    O --> P[Ready for Submission (Self-Contained)];

    style G fill:#f9f,stroke:#333,stroke-width:2px