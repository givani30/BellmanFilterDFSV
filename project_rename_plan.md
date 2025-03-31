# Project Renaming Plan: QF_Thesis to BellmanFilterDFSV

This document outlines the steps to rename the project from its current identification (e.g., `QF_Thesis`, `qf_thesis`) to `BellmanFilterDFSV`.

## Project Details

*   **Primary Language:** Python
*   **Key Libraries/Frameworks:** JAX, NumPy/SciPy
*   **Package Management/Build Tool:** `uv`, `setuptools` (via `pyproject.toml`)
*   **Version Control:** Git
*   **Documentation:** Sphinx
*   **Testing:** `pytest`
*   **Operating System:** Linux

## Renaming Workflow Diagram

```mermaid
graph TD
    A[Backup Project] --> B{Gather Info & Plan};
    B --> C[Rename Core Package];
    C --> D[Update Imports & Code Refs];
    D --> E[Update Config Files];
    E --> F[Update Build System];
    F --> G[Update Documentation];
    G --> H[Update Tests];
    H --> I[Update Scripts & Examples];
    I --> J[Rename Root Directory];
    J --> K[Update VCS];
    K --> L[Update CI/CD (if applicable)];
    L --> M[Update Environment];
    M --> N[Final Testing];
    N --> O[Update External Refs];

    subgraph Preparation
        A
        B
    end

    subgraph Code & Config Updates
        C
        D
        E
        F
        G
        H
        I
    end

    subgraph Infrastructure Updates
        J
        K
        L
        M
    end

    subgraph Verification & Finalization
        N
        O
    end
```

## Detailed Steps

1.  **Backup Project:**
    *   **Action:** Create a complete backup of the entire `/home/givanib/Documents/QF_Thesis` directory.
    *   **Reason:** Crucial safety step before making widespread changes.

2.  **Rename Core Package Directory:**
    *   **Action:** Rename the directory `src/qf_thesis` to `src/bellman_filter_dfsv`.
    *   **Reason:** Aligns the main source code directory with the new project name.

3.  **Update Source Code Imports and References:**
    *   **Action:** Perform a project-wide search and replace (case-sensitive and potentially case-insensitive variations) in all `.py` files (within `src/`, `tests/`, `examples/`, `scripts/`).
        *   Replace `import qf_thesis` with `import bellman_filter_dfsv`.
        *   Replace `from qf_thesis` with `from bellman_filter_dfsv`.
        *   Search for string literals or comments containing "qf_thesis" or "QF_Thesis" (or similar variations) and update them contextually to "BellmanFilterDFSV" or "bellman_filter_dfsv" as appropriate.
        *   Review class names, function names, variable names etc., especially in `__init__.py` files or core modules, to see if `QfThesis` or similar is part of the name and needs changing.
    *   **Reason:** Updates code to use the new package name and ensures consistency in naming conventions and documentation within the code.

4.  **Update Configuration Files:**
    *   **Action:** Edit `pyproject.toml`:
        *   Change the `name` field under `[project]` to `"bellman-filter-dfsv"` (standard PyPI naming convention - dashes instead of underscores).
        *   Update any paths referencing `src/qf_thesis` (e.g., package discovery settings).
        *   Review `[project.scripts]` or `[project.entry-points]` if they exist and update references.
    *   **Action:** Check `requirements.txt` if it exists and update any local package references (e.g., if using `-e .`).
    *   **Action:** Check `.coveragerc` for any path exclusions/inclusions referencing `qf_thesis`.
    *   **Reason:** Ensures build tools, dependency management, and other configurations recognize the new project name and structure.

5.  **Update Build System Artifacts:**
    *   **Action:** Delete the `src/qf_thesis.egg-info` directory. It will be regenerated with the new name upon the next build/install.
    *   **Reason:** Removes outdated build artifacts.

6.  **Update Documentation:**
    *   **Action:** Edit `docs/source/conf.py`:
        *   Change the `project` variable to `'BellmanFilterDFSV'`.
        *   Update `copyright` if it includes the old name.
        *   Review other variables like `html_title` or theme options.
    *   **Action:** Edit `README.md`: Update the main project title and any references to "QF_Thesis".
    *   **Action:** Search and replace "QF_Thesis" / "qf_thesis" (and variations) in all `.rst` files within `docs/source/`. Pay close attention to titles, references, and code examples.
    *   **Action:** Regenerate the documentation (e.g., navigate to `docs/` and run `make html` or the equivalent command).
    *   **Reason:** Ensures all user-facing and API documentation reflects the new project name.

7.  **Update Tests:**
    *   **Action:** Ensure all import paths in files under `tests/` have been updated (covered in step 3).
    *   **Action:** Check for any test configuration files (e.g., `pytest.ini`, or sections in `pyproject.toml`) that might reference the old package name or paths.
    *   **Action:** Run the full test suite to ensure all tests pass after the renaming.
    *   **Reason:** Verifies that the renaming process hasn't introduced regressions or broken test setups.

8.  **Update Scripts and Examples:**
    *   **Action:** Review files in `scripts/` and `examples/`. Ensure imports and any internal logic referencing the project name/package are updated (partially covered in step 3, but requires specific review).
    *   **Reason:** Ensures utility scripts and usage examples work correctly with the renamed package.

9.  **Rename Root Project Directory:**
    *   **Action:** Rename the main project directory `/home/givanib/Documents/QF_Thesis` to `/home/givanib/Documents/BellmanFilterDFSV`.
    *   **Reason:** Aligns the top-level project folder with the new name. *Note: This will change the Current Working Directory for future operations.*

10. **Update Version Control System (Git):**
    *   **Action:** Update the remote URL: `git remote set-url origin <new_repository_url>` (replace `<new_repository_url>` with the actual URL after renaming the repository on the hosting platform like GitHub/GitLab).
    *   **Action:** Rename the repository on the hosting platform itself (e.g., GitHub repository settings).
    *   **(Optional) Action:** Consider renaming key branches if they contain the old name (e.g., `git branch -m qf_thesis_feature new_feature`), although often `main` or `master` are sufficient.
    *   **Action:** Commit all the changes made so far with a clear commit message (e.g., "Rename project QF_Thesis to BellmanFilterDFSV"). Push the changes to the remote repository (including potentially renamed branches).
    *   **Reason:** Synchronizes the local repository, remote repository, and branch names with the new project identity.

11. **Update CI/CD Pipelines (If Applicable):**
    *   **Action:** Check for CI/CD configuration files (e.g., `.github/workflows/`, `.gitlab-ci.yml`, `Jenkinsfile`). Update repository names, paths, build commands, deployment steps, and any environment variables referencing the old name.
    *   **Reason:** Ensures automated build, test, and deployment processes function correctly after the rename.

12. **Update Development Environment:**
    *   **Action:** If using an editable install (`pip install -e .` or `uv pip install -e .`), uninstall the old package and reinstall with the new name from the renamed project directory:
        ```bash
        cd /home/givanib/Documents/BellmanFilterDFSV # Navigate to the NEW directory
        uv pip uninstall qf-thesis # Or the name found in pyproject.toml before changes
        uv venv .venv # Ensure venv exists or recreate if needed
        source .venv/bin/activate # Activate the venv (syntax might vary based on shell)
        uv pip install -e . # Install the renamed package
        ```
    *   **Reason:** Ensures the development environment correctly links to the renamed package.

13. **Final Testing and Verification:**
    *   **Action:** Run all tests again (`pytest` or equivalent).
    *   **Action:** Attempt to build the package (`uv pip install .` or build commands).
    *   **Action:** Run key examples or scripts from `examples/` and `scripts/`.
    *   **Action:** Build the documentation again.
    *   **Reason:** Comprehensive check to catch any remaining issues after all changes.

14. **Update External References:**
    *   **Action:** If the project is published (e.g., on PyPI) or referenced elsewhere (other projects, internal wikis, bookmarks), update those external references to the new name and repository URL.
    *   **Reason:** Ensures consistency and avoids broken links or references externally.