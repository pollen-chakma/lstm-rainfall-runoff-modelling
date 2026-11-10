# Daymet PCA10 versus PCA20: five-seed comparison

This experiment repeats the original Daymet models with seeds **42, 43, 44, 45 and 46**, paired across both setups. The two verified seed-42 runs are reused; eight new models are trained in separate folders. Original experiments are preserved.

| Setting | Both setups |
| --- | --- |
| Basins used for training | Same 1,326 donor IDs |
| Basins excluded from fitting and validation | Same 100 unseen IDs |
| Training years | 1980–1999 |
| Validation years | 2000–2004 |
| Test years, both basin groups | 2005–2009 |
| Meteorological inputs | prcp, tmax, tmin, srad, vp, dayl, pet |
| Input history | 365 days |
| LSTM | One layer, 64 hidden units |
| Epochs | 20, with 256 sampled windows per donor per epoch |
| Optimizer and batch | Adam 0.001; batch 128; gradient clipping 1 |
| Checkpoint for evaluation | Lowest validation loss within the 20 epochs |

PCA10 uses 20 static inputs: mean and standard deviation for each of 10 PCs. PCA20 uses 40 static inputs for 20 PCs. Thus, the input dimensions are 27 and 47, and the parameter counts are 23,873 and 28,993 respectively. Hidden-layer settings are unchanged. The PCA features are frozen across seeds; no PCA is refitted.

## Files and rerunning

- `PCA10/seed_43` through `seed_46`, and the corresponding PCA20 folders, contain a training notebook, a link to its complete input CSV, checkpoints, daily test predictions and basin performance CSVs. The seed-42 folders link to the completed original experiments.
- To train a model independently, copy its notebook and the actual CSV into a new empty folder. Select **Model A PyTorch (GPU)** and run all cells. A CSV link avoids duplication on Nibi; when moving to another computer, copy the underlying CSV file. Existing checkpoints in the same output folder cause the notebook to resume rather than restart training.
- `run_all.py` runs incomplete new notebooks sequentially and resumes interrupted runs from the last completed epoch. The lock prevents a second suite controller from running concurrently. Run it on a GPU compute node using the existing `Model_A/.venv/bin/python`, not on the login node.
- `suite_status.json`, individual `notebook_execution.log` files and `progress.html` show progress. Refresh the HTML page for current results.
- `run_manifest.json` records run paths, reused baselines and provenance. `protocol_equivalence.json` in each new run verifies that only the seed and output directory changed from its baseline code.

## Final analysis

After all runs finish, `analyse_multiseed.py` independently recalculates each basin's metrics from its daily predictions, checks the same IDs, test dates and observations across runs, and constructs the two ensembles by **averaging five daily discharge predictions before calculating metrics**.

The generated `Five_Seed_Comparison_Analysis.ipynb` and `multiseed_basin_analysis_dataset.csv` are sufficient to reproduce all comparison tables and plots. Copy these two files to an empty directory and run the notebook using **Basin Performance Diagnostics**, or Python with NumPy, SciPy, Matplotlib, Pillow and IPython. The notebook does not read the original model folders or daily prediction files.

`analysis_outputs` contains:

- `per_seed_group_performance.csv`: mean, median, quartiles and negative-score counts for each setup, seed, metric and Seen/Unseen group.
- `repeatability_summary.csv`: mean, sample standard deviation and range of the five seed medians; negative-count summaries.
- `paired_seed_comparison.csv`: paired differences in group medians. Wilcoxon tests and Holm-adjusted p-values are exploratory; five seeds provide limited power.
- `basin_improvement_consistency.csv` and `consistency_distribution.csv`: how often PCA20 improves each basin across the five paired seeds. Basins without evaluable discharge are identified by zero valid seed pairs.
- `ensemble_performance_summary.csv`: scores recalculated from ensemble discharge predictions.
- `PCA10_ensemble_daily_predictions.csv` and `PCA20_ensemble_daily_predictions.csv`: all daily ensemble predictions and observations, including missing observations.
- `hydrological_subgroup_comparison.csv` and `subgroup_definitions.json`: low-flow, intermittent and dry-basin comparisons, including group sizes and coverage requirements.
- `five_seed_report.html`, repeatability plots and basin-consistency plots.
- `verification.json` and `two_file_reproduction_verification.json`: numerical verification and an isolated notebook-plus-CSV reproduction check, including identical tables and figure pixels.

Seen-basin results test temporal generalization in training catchments. Unseen-basin results test predictions for catchments excluded from fitting. Nine seen basins lack test-period observations; their predictions remain exported, while unavailable scores remain missing.

Hydrological groups are descriptive and can overlap. Low flow uses the lower quartile of observed test-period mean discharge among adequately covered seen basins. Intermittent means at least 1% zero-flow observations; dry means test-period PET/precipitation greater than one. These diagnostics are used only for post-training analysis.

The original sampler uses `seed + epoch`, so adjacent seeds shift overlapping epoch-level window schedules. This tests repeatability of the original pipeline, including initialization changes; it does not create independent hydrological datasets. Differences between PCA10 and PCA20 do not by themselves isolate the effect of spatial variability from other differences in the static descriptors or input parameter count.
