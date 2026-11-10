# Poor basin performance analysis

The working folder on Nibi is `/home/pollen26/Desktop/LSTM_RR_Modelling/Poor_Basin_Performance_Analysis`.

To repeat the statistical analysis, keep these two project files together:

1. `Poor_Basin_Performance_Analysis.ipynb`
2. `basin_diagnostic_dataset.csv`

Open the notebook in JupyterLab, select **Basin Performance Diagnostics**, and run all cells in order. On another machine, select a Python environment with NumPy, pandas, SciPy, Matplotlib and IPython/Jupyter installed. The notebook's working directory must contain the CSV. No original CAMELS-SPAT files, model checkpoints, preparation scripts or previous analysis outputs are read by the notebook.

The input CSV contains 7,130 basin–experiment records (1,426 basins × five experiments), with 158 columns. Daily data were summarized once into this table. Repeating the notebook repeats the comparisons, classifications, tables and figures; it does not regenerate daily statistics from the raw archive. Change the relevant notebook definitions to explore alternative thresholds.

The notebook regenerates `analysis_outputs/`, including:

- `negative_basin_catalog.csv`: exact basin IDs with negative NSE or KGE, metrics and diagnostic characteristics.
- `failure_summary_by_experiment_seen_unseen.csv`: model-level failure counts and score summaries.
- `persistent_failures_2005_2009.csv`: failures repeated across the four same-period models.
- `feature_group_contrasts.csv`: negative versus nonnegative basin characteristics, effect sizes and exploratory multiple-testing-adjusted comparisons.
- `failure_rates_by_class.csv`: class-specific rates, sample sizes and uncertainty intervals.
- `binary_class_risk_comparisons.csv`: comparisons of hydrological indicator groups.
- `continuous_score_associations.csv`: feature–score associations within model and Seen/Unseen strata.
- `negative_KGE_error_components.csv`: correlation, variability and volume-bias diagnostics.
- `coverage_sensitivity.csv` and `class_rates_adequate_coverage.csv`: sensitivity to observation coverage.
- `physical_range_quality_flags.csv`: invalid physical descriptors excluded from comparisons; basins retained.
- Three diagnostic figures, `FINDINGS.md`, data coverage and provenance records.

The Daymet + PCA10 analysis has 84 negative-NSE and 111 negative-KGE Seen basins (of 1,317 scored), and eight negative-NSE and 14 negative-KGE Unseen basins (of 100). This is post-hoc evidence of associations, not causal attribution. Negative NSE and negative KGE do not share the same benchmark meaning; the notebook also reports KGE below 1−sqrt(2).

The four 2005–2009 experiments are comparable in test years. Extended ERA5 tests on 2015–2023-01-03 and is analyzed separately. No karst/limestone, abstraction-operation or gauge-rating assessment is inferred from permeability or reservoir area alone. Class definitions are study-specific observational proxies.

The preparation and environment scripts are audit helpers, not required analysis inputs. Do not rerun the one-time CSV preparation unless deliberately rebuilding the input dataset.

Verification complete: the notebook was rerun in a fresh directory containing only the notebook and CSV. All eight code cells completed. Fourteen output CSV tables and both findings/completion reports matched, using numerical tolerances rtol=1e-10 and atol=1e-12. See two_file_reproduction_verification.json. The permanent analysis environment passes all required imports.
