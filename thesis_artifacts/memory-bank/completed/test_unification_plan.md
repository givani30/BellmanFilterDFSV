# Test Framework Unification and Enhancement Plan

**Date:** 2025-06-04

**Goal:** Standardize the testing framework using `pytest`, create unified tests for core filter functionalities (stability, log-likelihood), and add specific tests to verify parameter constraints (positivity, PSD, stationarity for diagonal matrices) after untransformation.

**Phase 1: Framework Standardization and Unified Test Setup**

1.  **Standardize on `pytest`:**
    *   Ensure `pytest` and necessary plugins (like `pytest-cov`) are in `pyproject.toml` dev dependencies.
    *   Configure `pytest` settings if needed.
2.  **Create Unified Test File:**
    *   Create `tests/test_unified_filters.py`.
3.  **Develop Common Fixtures (in `tests/conftest.py` or `tests/test_unified_filters.py`):**
    *   `params_fixture(N, K)`: Generates `DFSVParamsDataclass`.
    *   `data_fixture(params, T, seed)`: Simulates DFSV data.
    *   `filter_instances_fixture(params)`: Instantiates and returns BF, BIF, PF instances.

**Phase 2: Implement Unified Tests**

1.  **Implement Tests in `tests/test_unified_filters.py`:**
    *   Use `pytest.mark.parametrize` to iterate through filter instances.
    *   **`test_filter_stability(filter_instance, params, observations)`:**
        *   Run the filter's primary filtering method (`filter_scan` for BF/BIF, `filter` for PF).
        *   Assert finite states and covariances/information matrices.
    *   **`test_log_likelihood_wrt_params(filter_instance, params, observations)`:**
        *   Call `filter_instance.log_likelihood_wrt_params(params, observations)`.
        *   Assert finite scalar log-likelihood.

**Phase 3: Refactor Existing Tests**

1.  **Convert `unittest` to `pytest`:**
    *   Refactor `tests/test_bellman_unified.py`, `tests/test_particle_filter.py`, `tests/test_transformations.py`.
    *   Replace `setUp` with fixtures, `self.assert*` with `assert`.
2.  **Integrate Unified Tests:**
    *   Remove redundant stability/log-likelihood tests from original files.
3.  **Retain/Adapt Specific Tests:**
    *   Keep filter-specific tests (e.g., BIF vs BF comparison, PF smoother accuracy).
    *   Adapt to use common fixtures.
    *   Consider renaming `test_bellman_unified.py` to `test_bellman.py`.

**Phase 4: Enhance Transformation Constraint Tests**

1.  **Add Constraint Verification Tests in `tests/test_transformations.py`:**
    *   Create `test_untransformed_parameter_properties`.
    *   Perform round-trip transformation (`transform_params` -> `untransform_params`).
    *   Verify properties of `untransformed` parameters:
        *   `sigma2`: Assert diagonal elements > 0.
        *   `Q_h`: Assert diagonal elements >= 0 (PSD for diagonal).
        *   `Phi_f`, `Phi_h`: Assert absolute diagonal elements < 1.0 (Stationarity for diagonal).

**Phase 5: Finalization**

1.  **Run Full Test Suite:** `uv run pytest tests/`.
2.  **Code Formatting/Linting:** Adhere to PEP 8.
3.  **Documentation (Optional):** Update testing structure docs.
4.  **Memory Bank Update:** Add entry to `progress.md`.

**Visual Representation (Mermaid):**

```mermaid
graph TD
    subgraph Fixtures [pytest Fixtures (e.g., conftest.py)]
        F1[Fixture: params_fixture] --> P(DFSVParams)
        F2[Fixture: data_fixture] --> D(Observations)
        F3[Fixture: filter_instances_fixture] --> Filters(BF, BIF, PF Instances)
    end

    subgraph Unified Tests (test_unified_filters.py)
        P --> UT1 & UT2
        D --> UT1 & UT2
        Filters -- @pytest.mark.parametrize --> UT1(test_filter_stability)
        Filters -- @pytest.mark.parametrize --> UT2(test_log_likelihood_wrt_params)
    end

    subgraph Specific Tests (Refactored to pytest)
        ST_BF[test_bellman.py]
        ST_BIF[test_bellman_information.py]
        ST_PF[test_particle_filter.py]
        ST_T[test_transformations.py]
    end

    subgraph Transformation Constraint Tests (in test_transformations.py)
        P --> TCT1(test_untransformed_parameter_properties)
        TCT1 -- Checks --> Constraints{Stationarity, PSD, Positivity}
    end

    Fixtures --> ST_BF
    Fixtures --> ST_BIF
    Fixtures --> ST_PF
    Fixtures --> ST_T

    ST_T --> TCT1