# Comprehensive Plan for Thesis Completion (Deadline: April 30th, 2025) - v3.4

**Overall Goal:** Complete the implementation, evaluation, and documentation of the Bellman Information Filter (BIF) for the specified DFSV model, compare it against benchmarks (Particle Filter), evaluate hyperparameter estimation performance, and apply it to real financial data, culminating in the final thesis document by **April 30th, 2025**.

**Current State:** (As of 2025-11-04)

* Core BIF and PF implementations are functional and stabilized.
* Test framework is robust and standardized using pytest.
* **`mu` identifiability investigation complete:** Decision made to **fix `mu`** in BIF estimations.
* Initial simulation runs (known parameters) for BF/PF/BIF are complete.
* Optimization framework enhanced with centralized optimizer creation, parameter logging, and improved performance options.
* `mu` fixing logic aligned across all optimization utilities.
* **Next major focus:** Completing hyperparameter estimation study, real data preparation, real data application, and thesis writing within April 2025.

**Phase 1: Foundational Analysis & Initial Simulations**

1. **Task: Deep Dive into `mu` Identifiability (Critical Path) - COMPLETED**
    * **Objective:** Understand `mu` estimation difficulty and determine robust strategy.
    * **Sub-tasks:** Review existing results, profile likelihood, experiment with priors/fixing/restrictions, document strategy.
    * **Deliverable:** Documented findings and decision to fix `mu`. (`decisionLog.md` [2025-06-04])
    * **Sub-task:** experiment with diagonal phi_f (2025-06-04)

2. **Task: Execute Initial BIF Simulation Runs (Known Params) - COMPLETED**
    * **Objective:** Generate BIF simulation results (known parameters) for comparison.
    * **Sub-tasks:** Configure script, execute runs, organize outputs.
    * **Deliverable:** Raw BIF simulation output files.

3. **Task: Analyze Existing BF/PF Simulation Results (Known Params) (Parallel)**
    * **Objective:** Begin processing benchmark filter results (known parameters).
    * **Sub-tasks:** Develop/refine analysis scripts, calculate metrics, generate preliminary BF/PF tables/plots.
    * **Deliverable:** Analysis scripts, preliminary BF/PF results.

4. **Task: Finalize Real Data Preparation (Parallel)**
    * **Objective:** Ensure real-world datasets are ready.
    * **Sub-tasks:** Confirm cleaning/alignment, perform final EDA.
    * **Deliverable:** Cleaned data files.

5. **Task: Initiate Thesis Writing (Parallel)**
    * **Objective:** Start drafting core sections.
    * **Sub-tasks:** Draft Intro, Lit Review, Methodology (initial), Data chapters.
    * **Deliverable:** Initial drafts of Chapters 1, 2, 4, partial 3.

**Phase 2a: Initial Simulation Analysis (Apr 4-8)**

6. **Task: Analyze BIF Simulation Results (Known Params) & Comparative Analysis**
    * **Objective:** Evaluate BIF performance (known params) and compare filters.
    * **Sub-tasks:** Process BIF outputs (fixed `mu`), calculate metrics (accuracy, speed, stability), perform comparative analysis (BIF vs. BF vs. PF).
    * **Deliverable:** Initial simulation analysis, comparative tables/plots, draft updates for Simulation Results chapter.

**Phase 2b: Hyperparameter Estimation Study (Apr 9-19)**

7. **Task: Design Hyperparameter Estimation Study Configurations**
    * **Objective:** Define configurations for estimation tests.
    * **Sub-tasks:** Review literature/goals, define simulation settings, document configurations.
    * **Deliverable:** Documented simulation configurations.

8. **Task: Implement Parameter Logging during Optimization**
    * **Objective:** Enhance scripts to log parameter convergence.
    * **Sub-tasks:** Modify optimization wrappers/callbacks, define logging format/destination.
    * **Deliverable:** Updated optimization scripts.

9. **Task: Test PF Hyperparameter Estimation Suite**
    * **Objective:** Ensure PF estimation setup works.
    * **Sub-tasks:** Run small-scale PF estimation tests, debug issues.
    * **Deliverable:** Confirmed working PF estimation script.

10. **Task: Adapt/Create Google Cloud Batch Script for Estimation Study**
    * **Objective:** Prepare for cloud runs (if needed).
    * **Sub-tasks:** Leverage existing batch code, adapt for estimation runs (PF & BIF w/ fixed `mu`), handle inputs/outputs.
    * **Deliverable:** Functional Batch script/template.

11. **Task: Run Hyperparameter Estimation Simulations**
    * **Objective:** Execute estimation runs (PF & BIF w/ fixed `mu`).
    * **Sub-tasks:** Launch batch jobs or run locally, monitor progress, collect results/logs.
    * **Deliverable:** Raw estimation simulation results/logs.

12. **Task: Analyze Hyperparameter Estimation Results**
    * **Objective:** Evaluate estimation performance.
    * **Sub-tasks:** Develop/run analysis scripts, generate tables/plots (bias, RMSE, convergence steps).
    * **Deliverable:** Estimation analysis, results tables/plots.

13. **Task: Integrate Estimation Results into Thesis**
    * **Objective:** Incorporate findings into thesis draft.
    * **Sub-tasks:** Draft/update relevant sections (Methodology, Simulation Results).
    * **Deliverable:** Updated thesis draft sections.

**Phase 3: Real Data Application & Evaluation (Apr 22-25)**

14. **Task: Apply Filters to Real Data**
    * **Objective:** Estimate models using real data (BIF w/ fixed `mu`).
    * **Sub-tasks:** Adapt scripts, run BIF (fixed `mu`), run benchmarks (PF/BF), extract results.
    * **Deliverable:** Real data estimation scripts, saved estimates/states.

15. **Task: Evaluate Real Data Performance**
    * **Objective:** Assess practical performance (MSPE, Sharpe).
    * **Sub-tasks:** Implement/run predictive evaluation, implement/run portfolio evaluation.
    * **Deliverable:** Performance evaluation scripts, real data results tables/plots.

**Phase 4: Finalization & Submission (Parallel & Apr 26-30)**

16. **Task: Continue Thesis Writing (Parallel)**
    * **Objective:** Integrate all findings.
    * **Sub-tasks:** Finalize Methodology (fixed `mu`), draft/finalize Simulation Results (both studies), draft/finalize Empirical Results, draft Benchmarking Analysis.
    * **Deliverable:** Near-complete draft chapters.

17. **Task: Synthesize Findings & Write Conclusion**
    * **Objective:** Summarize results and contributions.
    * **Sub-tasks:** Review all results, write Conclusion chapter, write Abstract.
    * **Deliverable:** Draft Conclusion and Abstract.

18. **Task: Compile Full Thesis Draft & Refine**
    * **Objective:** Assemble and polish document.
    * **Sub-tasks:** Integrate all content, ensure consistency, refine narrative, check formatting, finalize references.
    * **Deliverable:** Complete first draft.

19. **Task: Review Cycles & Final Submission**
    * **Objective:** Incorporate feedback and submit by deadline.
    * **Sub-tasks:** Supervisor feedback loop (rapid), incorporate changes, final proofreading, prepare submission PDF, submit by 2025-04-30.
    * **Deliverable:** Final, submitted thesis.

**Timeline Visualization (for April 30th, 2025 Deadline)**

```mermaid
%%{ init: { 'theme': 'base' } }%%
gantt
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m-%d
    title       Thesis Plan v3.4 (Deadline: 2025-04-30)
    excludes    weekends
    todayMarker stroke-width:3px,stroke:#0000FF,opacity:0.7

    section Phase 1 (Completed by 2025-04-04)
    Investigate mu ID        :done, P1_mu, 2025-04-01, 3d
    Run Initial BIF Sims     :done, P1_bifsim, 2025-04-01, 3d

    section Phase 2a: Initial Sim Analysis (Apr 4-8)
    Analyze BF/PF Sims (Init): P1_anabfpf, 2025-04-04, 2d
    Analyze BIF Sims (Init)  : crit, P2a_anabif, after P1_anabfpf, 2d
    Compare Initial Sims     : crit, P2a_simcomp, after P2a_anabif, 1d

    section Phase 2b: Hyperparam Estimation Study (Apr 9-19)
    Design Est. Configs      : P2b_des, 2025-04-09, 1d
    Implement Param Logging  : P2b_log, after P2b_des, 1d
    Test PF Est. Suite       : P2b_testpf, after P2b_log, 1d
    Adapt Cloud Batch Script : P2b_batch, after P2b_testpf, 1d
    Run Estimation Sims      : crit, P2b_run, after P2b_batch, 3d // Very compressed
    Analyze Estimation Results: crit, P2b_anaest, after P2b_run, 2d
    Integrate Est. Results   : P2b_write, after P2b_anaest, 1d // Integrate into writing later

    section Phase 3: Real Data App & Eval (Apr 22-25)
    Finalize Real Data Prep  : P1_dataprep, 2025-04-04, 10d // Parallel
    Run on Real Data         : crit, P3_realapp, 2025-04-22, 2d // Assumes data ready
    Evaluate Real Data Perf. : crit, P3_realeval, after P3_realapp, 2d

    section Phase 4: Finalization (Parallel & Apr 26-30)
    Write Sim/Empirical Chaps: crit, P4_write2, 2025-04-04, 15d // Parallel writing
    Write Conclusion/Abs.    : crit, P4_write3, 2025-04-26, 1d
    Compile & Refine Draft   : crit, P4_compile, 2025-04-27, 1d
    Supervisor Review        : crit, P4_review, 2025-04-28, 1d // Assumes rapid review
    Final Revisions/Submit   : crit, P4_submit, 2025-04-29, 2d // Submit on 30th
