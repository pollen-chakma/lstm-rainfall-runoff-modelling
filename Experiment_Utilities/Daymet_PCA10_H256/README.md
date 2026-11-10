# Daymet + PCA10 mean/std — H256, dropout 0.2, adaptive max50

The new Nibi experiment lives in:

`Desktop/LSTM_RR_Modelling/Model A Lumped (Daymet + PCA10 H256 Dropout02 Adaptive50)`

Only these two input files are needed to train, evaluate, and compare again:

- `Daymet_PCA10_H256_Dropout02_Adaptive50.ipynb`
- `daymet_pca10_discharge_1980_2009.csv`

Open the notebook in a GPU JupyterLab session, select **Model A PyTorch (GPU)**, and run cells in order. Python needs NumPy, PyTorch, IPython, and a Jupyter kernel. Baseline comparison statistics and reference preprocessing values are embedded in the notebook; they do not enter the LSTM inputs. No other experiment files are required to rerun. In the original workspace, protected baseline file hashes are additionally checked if those files are present.

The CSV is a regular, independent copy of the original PCA10 mean/std CSV, verified by SHA-256. Original schema, IDs, data order, masks, 1980–1999/2000–2004/2005–2009 splits, sample counts, loss, preprocessing and batch sizes remain unchanged. There are seven meteorological inputs and twenty static descriptors (10 PC means and 10 PC standard deviations), 27 input channels in total.

The requested changes are one layer with 256 hidden units, 0.2 dropout on the final LSTM output before the linear head, up to 50 epochs, LR 0.001 halved after three consecutive non-improving validation epochs (minimum 0.00001), and early stopping after ten consecutive non-improving validation epochs. Improvement means strictly lower validation loss. The best validation checkpoint is selected, and only then is the test period evaluated. One seed, 42, is used.

`outputs/results_seed42/last.pt` saves completed epoch state, optimizer, scheduler counters, early-stopping counters, full history and all RNG states. Rerun the notebook to resume. Partial epochs are restarted from the previous completed epoch. Copy the two inputs into a new empty folder to start fresh. Do not run another training process on the same output folder while the first is active; a file lock prevents concurrent runs.

The full-batch GPU check uses training batch 128 and validation batch 256 in float32. Memory use is saved in `outputs/gpu_memory_check.json`. Training will stop on an actual memory error rather than silently changing the requested batch or precision.

Outputs include:

- `basin_model_performance.csv`: all 1,426 basin rows, seen/unseen labels, NSE, KGE, RMSE, MAE, volume bias, mean bias, low/high-flow thresholds, sample counts and errors.
- `baseline_seed42_basin_performance.csv`: immutable baseline reference, including matching flow diagnostics.
- `basin_baseline_comparison.csv`: baseline/new metrics and paired changes by basin.
- `model_performance_summary.csv`: distributions by model, seen/unseen and metric, including negative NSE/KGE counts.
- `paired_comparison_summary.csv`: common-basin improvement/worsening counts and paired changes.
- `training_validation_loss.svg`, `model_report.html`: training curve and comparison report.
- `results_seed42/`: checkpoints, epoch history, selected epoch, preprocessing and per-basin daily prediction CSVs in `unseen100/` and `donor_temporal/`.
- Verification JSON files, `completion.json`, and refreshable `progress.html`.

Low flow is observed test discharge at or below the basin's Q10; high flow is at or above Q90. Thresholds are calculated after checkpoint selection from the finite observed/predicted pairs in 2005–2009. Ties are included, so groups may exceed 10%. These test-derived thresholds are descriptive evaluation strata, not model features or fitted normalization. Both experiments use identical dates and thresholds. Errors are RMSE, MAE, mean bias (mm/day), and volume bias (%); percent bias is undefined when observed subset volume is zero. Negative-prediction diagnostics count only scored daily pairs. Missing observations are never replaced by zero.

This single-seed comparison tests the combined architecture and training change. It does not isolate which change caused a difference or establish repeatability. Existing spatial-overlap limitations of the split are retained.
