# Refined Plan v2: Preparing `thesis_code_submission.zip` (Under 50MB)

**Goal:** Create a ZIP archive (`thesis_code_submission.zip`) containing the minimal, essential components required to install the `bellman_filter_dfsv` package and reproduce the key figures and tables from Chapter 5 of the thesis using provided scripts and *curated saved output data* sourced from the `final_out` directory. The archive must be under 50MB. Full code, simulation results, and all generated outputs will reside in the linked GitHub repository.

**Phase 1: Information Gathering & Assessment**

1.  **List `thesis_code_submission` Contents:** Confirm the current state of the target directory.
2.  **List `final_out` Contents:** Investigate the `final_out` directory (in the workspace root) to identify curated results.
3.  **Analyze `final_out/empirical/insample/`:** Mentally review the listing from step 2. Identify potential data files (`.json`, `.csv`, `.pkl`, etc.) within this subdirectory that likely contain the saved states, parameters, metrics, and residual analysis results needed to generate Chapter 5 figures and tables. Note the presence of figures (`.png`, `.pdf`) and tables (`.tex`) which should *not* be included in the ZIP.

**Phase 2: Identify Core Code & Script Components**

1.  **Core Package:** Verify `thesis_code_submission/src/bellman_filter_dfsv/` contains the necessary Python modules.
2.  **Installation Files:** Confirm `thesis_code_submission/requirements.txt` and `thesis_code_submission/setup.py` are present.
3.  **Input Data:** Confirm `thesis_code_submission/Data/processed/VW_Returns_Monthly.csv` is present. Check its size.
4.  **Reproduction Scripts:**
    *   Identify the *specific* Python scripts within `thesis_code_submission/scripts/empirical/` required to generate Chapter 5 figures/tables *from saved data* (e.g., `combined_model_metrics.py`, `01_initial_residual_analysis.py`, `02_garch_analysis.py`, potentially others mentioned in the README or known to be essential).
    *   *Action (Requires Analysis):* Check the import statements of these key empirical scripts. Identify any necessary utility functions/modules imported from outside `scripts/empirical/` (e.g., `scripts/utils/`, `scripts/simstudy_2/analysis_plotting_utils.py` if used). These utility scripts must be retained.
5.  **Examples:** Select a minimal set of scripts from `thesis_code_submission/examples/` (e.g., `simple_dfsv_example.py`) to demonstrate basic usage. Flag others for potential removal to save space.

**Phase 3: Select and Stage Required Empirical Outputs**

1.  **Source Selection:** Use `final_out/empirical/insample/` as the definitive source for saved empirical *data*.
2.  **Minimal Data Identification:** Based on the needs of the reproduction scripts identified in Phase 2, select the *absolute minimum* set of data files (`.json`, `.csv`, `.pkl`, etc.) from `final_out/empirical/insample/` required to run those scripts. **Crucially, exclude all pre-generated figures (.png, .pdf) and tables (.tex).**
3.  **Staging Location:** Plan to create `thesis_code_submission/outputs/empirical/`.
4.  **Staging Action:** Plan to copy *only* the selected minimal data files from `final_out/empirical/insample/` into `thesis_code_submission/outputs/empirical/`.

**Phase 4: Plan Code Submission Folder Cleanup**

1.  **Redundant/Large Files:**
    *   Flag `thesis_code_submission/thesis_code.zip` for deletion.
    *   Flag `thesis_code_submission/batch_outputs/` for deletion.
    *   Flag the original `thesis_code_submission/outputs/` directory (if it exists and contains non-empirical or duplicated data) for deletion.
2.  **Non-Essential Scripts:**
    *   Flag all scripts in `thesis_code_submission/scripts/` *except* for the essential empirical reproduction scripts and their identified utilities (from Phase 2). This includes simulation study scripts (`simstudy_1/`, `simstudy_2/`, `simulation/`), optimization scripts (`optimization/`), general analysis scripts (`analysis/`), data processing scripts (`data_processing/` - unless needed by empirical scripts), and other miscellaneous scripts.
3.  **Non-Essential Examples:** Flag non-selected example scripts for deletion.

**Phase 5: Refine `README.md`**

1.  **Content Update:**
    *   Clearly state the package reproduces Chapter 5 results *from saved data* located in `outputs/empirical/`.
    *   Emphasize the 50MB limit and direct users to the GitHub repo for full code, simulation scripts, all outputs (including those excluded from the ZIP), and the thesis PDF.
    *   Add a specific note stating that large output files (e.g., full simulation results, potentially some large empirical `.pkl` files if pruned in Phase 6) are available on GitHub or upon request.
    *   Ensure usage instructions correctly reference the included empirical scripts and the `outputs/empirical/` data path.
    *   Update the folder structure description to reflect the final, cleaned structure.
    *   Verify the GitHub link: `https://github.com/givani30/BellmanFilterDFSV`.

**Phase 6: Execute Cleanup, Assemble & Size Check (Iterative)**

1.  **Execute Deletions:** Remove the flagged files and directories from `thesis_code_submission` (as planned in Phase 4).
2.  **Execute Staging:** Copy the selected minimal empirical data files from `final_out/empirical/insample/` to `thesis_code_submission/outputs/empirical/` (as planned in Phase 3).
3.  **Assemble:** Ensure all retained components (`src`, `requirements.txt`, `setup.py`, selected `examples`, essential `scripts/empirical` + utils, `Data/processed`, staged `outputs/empirical`) are correctly placed.
4.  **Check Size:** Determine the total size of the cleaned `thesis_code_submission` folder.
5.  **Prune if Necessary (> 50MB):**
    *   Identify largest remaining files/folders (likely `outputs/empirical/` or `src/`).
    *   *Priority 1:* Remove remaining non-essential examples.
    *   *Priority 2:* Remove the largest *data* files from `outputs/empirical/`, prioritizing those least critical for core Chapter 5 results. Document these removals carefully in the README.
    *   *Priority 3:* Check `Data/processed/VW_Returns_Monthly.csv` size. If excessive, consider noting it's available on GitHub/request and removing it (though ideally, keep it if possible).
    *   Re-check size. Repeat pruning if needed.

**Phase 7: Finalization**

1.  **Final Review:** Perform a thorough check of the final folder contents, script paths (especially data loading paths in empirical scripts), and the updated `README.md`.
2.  **Create ZIP:** Create the final `thesis_code_submission.zip` archive.

**Mermaid Diagram (Refined Plan v2):**

```mermaid
graph TD
    subgraph Phase 1: Gather & Assess
        A[List thesis_code_submission] --> C
        B[List final_out] --> C{Review Listings}
        C --> C1[Assess final_out/empirical/ for data files (JSON, CSV, PKL)]
    end

    subgraph Phase 2: Identify Core Code/Scripts
        D[Verify src/, reqs.txt, setup.py] --> E
        E[Check Data/processed/ size] --> F
        F[Identify essential empirical scripts] --> G
        G[Check script imports for utils] --> H
        H[Select minimal examples] --> I[Core Components Identified]
    end

    subgraph Phase 3: Select & Stage Outputs
        J[Source from final_out/empirical/insample/] --> K
        K[Select MINIMAL data files (NO figures/tables)] --> L
        L[Plan: Create outputs/empirical/ in submission folder] --> M
        M[Plan: Copy selected data files to outputs/empirical/]
    end

    subgraph Phase 4: Plan Cleanup
        N[Flag thesis_code.zip for deletion] --> O
        O[Flag batch_outputs/ for deletion] --> P
        P[Flag original outputs/ for deletion] --> Q
        Q[Flag non-essential scripts (sim*, optim*, analysis*, etc.)] --> R
        R[Flag non-selected examples for deletion]
    end

    subgraph Phase 5: Update Docs
        S[Refine README.md: Scope, Size, GitHub, Usage, Structure, Exclusions]
    end

    subgraph Phase 6: Execute, Assemble, Size Check
        T[Execute Deletions (Phase 4)] --> U
        U[Execute Staging (Phase 3)] --> V
        V[Assemble retained components] --> W{Check Size}
        W -- > 50MB --> X[Identify Large Items]
        X --> Y{Prune: Examples? Output Data? Input Data?}
        Y --> Z[Remove items, Update README]
        Z --> W
        W -- <= 50MB --> AA[Size OK]
    end

    subgraph Phase 7: Finalize
        BB[Final Review: Contents, Paths, README] --> CC[Create thesis_code_submission.zip]
    end

    C1 --> K
    I --> V
    R --> T
    M --> U
    S --> BB
    AA --> BB