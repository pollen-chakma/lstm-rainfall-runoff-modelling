# Model A file guide and Jupyter workflow

## Notebooks
View the full-record model with Model_A_Lumped_ERA5_1950_2023.ipynb. Run All refreshes status/results and does not train.
Train and test again with Run_Full_Training_and_Test.ipynb in a GPU session. ACTION=inspect starts no model. Change to ACTION=new_run and Run All to create a separate timestamped folder. The original inputs/results are preserved.

## Periods and shared dependencies
Daymet and original ERA5: train 1980–1999, validate 2000–2004, test 2005–2009.
Full ERA5: train 1950–2009, validate 2010–2014, test 2015–latest available early-2023 data. Individual record lengths vary.
Keep the sibling Model_A folder: .venv/bin/python supplies dependencies, train_model_a.py supplies ModelA and scores, prepare_daymet.py supplies source-reading helpers.
All named experiments retain 1326/100 IDs, seven dynamic inputs, five attributes, lookback 365, hidden size 64, seed 42, 20 epochs, batch 128, Adam 0.001, gradient clip 1, 256 sampled windows/donor/epoch and up to 128 fixed validation windows/donor. Not every historical day is sampled each epoch.
Dynamic inputs: prcp, tmax, tmin, srad, vp, dayl, pet. Static inputs: basin_area, elev_mean, slope_mean, stream_density, elongation_ratio. No PCA or discharge input.

## Top-level inventory

### Model A Lumped (Daymet)

| File or folder | Purpose |
|---|---|
| .ipynb_checkpoints | Jupyter autosave checkpoints. |
| Model_A_Temporal_Test.ipynb | Original or earlier temporal test report. Prefer the explicitly named ERA5 report in the ERA5 folder. |
| Model_A_Unseen100.ipynb | Original Daymet monitoring/results notebook. Run All reads results, not training. |
| __pycache__ | Automatic Python import cache. |
| coverage_audit.csv | Source checks and eligible train/validation/test windows. Full-record version also lists observed coverage dates/counts. |
| evaluate_temporal.py | Test the frozen best model on donor catchments during 2005–2009 without retraining. |
| experiment_config.json | Input names, source paths, date cutoffs, quality rules and exact donor/holdout IDs. |
| fast_dates.py | Decode and align daily NetCDF time bounds, including streamflow observations. |
| final_verification.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| inspect_experiment.py | Render original experiment status, configuration, losses and results for report notebooks. |
| orchestrator.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| orchestrator_process.json | Historical PID/launch metadata. Not proof of a currently running job in a new session. |
| preparation.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| preparation_2worker_20261031_220556.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| preparation_before_optimization_20261031_220014.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| preparation_process.json | Historical PID/launch metadata. Not proof of a currently running job in a new session. |
| prepare_all.py | Original Daymet preparation, coverage audit and fixed split creation. |
| prepared | One NPZ per catchment: dates, x[days,7] meteorology, y[days] observed discharge in mm/day, static[5]. NaN means missing. Full-record caches also contain audit metadata. |
| run_experiment.py | Original Daymet pipeline driver. Use isolated reruns instead of running in a completed result folder. |
| run_seed42 | Model checkpoints, fitted preprocessing, loss history and all test outputs for seed 42. |
| shared_polygon_overlap_screen.csv | Donor/holdout pairs sharing polygon IDs; not an exhaustive geometry audit. |
| split_manifest.csv | One row per basin: identity, country/group/climate proxy, fixed split and eligible window counts. |
| status.json | Preparation/training or unseen-test stage status; does not necessarily mean later temporal testing is finished. |
| stratum_counts.csv | Fixed catchment counts by split, basin group, climate proxy and country. |
| summarize_unseen.py | Postprocess unseen results and shared-polygon screen summaries; no training. |
| temporal_evaluation.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| train_unseen.py | Fit donor-only preprocessing, train the shared GPU LSTM, select the validation checkpoint and evaluate unseen catchments. |
| training.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| training_basins.txt | Exact 1326 training donor IDs. |
| training_process.json | Historical PID/launch metadata. Not proof of a currently running job in a new session. |
| unseen_test_basins.txt | Exact 100 held-out IDs, excluded from fitting and checkpoint selection. |
| verification.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| verify_protocol.py | Checks donor-only fitting, imputation, equal-basin sampling, temporal boundaries and unseen-label independence. |

### Model A Lumped (ERA5)

| File or folder | Purpose |
|---|---|
| .ipynb_checkpoints | Jupyter autosave checkpoints. |
| Model_A_Lumped_ERA5.ipynb | Original ERA5 monitoring/results notebook and Daymet comparison. Run All does not train. |
| Model_A_Lumped_ERA5_Temporal_Test.ipynb | Explicitly named original ERA5 temporal test report. |
| Model_A_Temporal_Test.ipynb | Original or earlier temporal test report. Prefer the explicitly named ERA5 report in the ERA5 folder. |
| README.md | Experiment overview and usage. |
| __pycache__ | Automatic Python import cache. |
| comparison.html | Rendered original forcing comparison. |
| completion.json | Whole-pipeline completion marker, coverage and test summaries. |
| coverage_audit.csv | Source checks and eligible train/validation/test windows. Full-record version also lists observed coverage dates/counts. |
| daymet_era5_comparison.csv | Original 2005–2009 test medians for Daymet versus ERA5. |
| daymet_preservation_hashes.json | Protected Daymet hashes used during the original ERA5 experiment. |
| evaluate_temporal.py | Test the frozen best model on donor catchments during 2005–2009 without retraining. |
| experiment_config.json | Input names, source paths, date cutoffs, quality rules and exact donor/holdout IDs. |
| fast_dates.py | Decode and align daily NetCDF time bounds, including streamflow observations. |
| final_verification.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| finalize_era5_reports.py | Generate short-period ERA5 reports and Daymet/ERA5 comparison. |
| independent_aggregation_verification.txt | Independent raw-hourly check of ERA5 daily aggregation. |
| inspect_experiment.py | Render original experiment status, configuration, losses and results for report notebooks. |
| paired_protocol_audit.json | Saved paired-experiment verification result. |
| paired_protocol_audit.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| pipeline.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| preparation.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| prepare_era5.py | Original 1980–2009 ERA5 hourly-to-daily conversion with the fixed Daymet split and attributes. |
| prepared | One NPZ per catchment: dates, x[days,7] meteorology, y[days] observed discharge in mm/day, static[5]. NaN means missing. Full-record caches also contain audit metadata. |
| report_generation.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| reports_complete.json | Report generation completion marker. |
| run_all_era5.py | Short-period ERA5 pipeline: preparation checks, training, both tests and preserved-Daymet comparison. |
| run_seed42 | Model checkpoints, fitted preprocessing, loss history and all test outputs for seed 42. |
| shared_polygon_overlap_screen.csv | Donor/holdout pairs sharing polygon IDs; not an exhaustive geometry audit. |
| source_snapshot | Copies of run source/configuration and SHA-256 hashes for reproducibility. |
| split_manifest.csv | One row per basin: identity, country/group/climate proxy, fixed split and eligible window counts. |
| status.json | Preparation/training or unseen-test stage status; does not necessarily mean later temporal testing is finished. |
| stratum_counts.csv | Fixed catchment counts by split, basin group, climate proxy and country. |
| summarize_unseen.py | Postprocess unseen results and shared-polygon screen summaries; no training. |
| temporal_evaluation.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| train_unseen.py | Fit donor-only preprocessing, train the shared GPU LSTM, select the validation checkpoint and evaluate unseen catchments. |
| training.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| training_basins.txt | Exact 1326 training donor IDs. |
| unseen_test_basins.txt | Exact 100 held-out IDs, excluded from fitting and checkpoint selection. |
| verification.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| verify_paired_experiments.py | Audit unchanged IDs, shared data and model settings across original forcing experiments. |
| verify_protocol.py | Checks donor-only fitting, imputation, equal-basin sampling, temporal boundaries and unseen-label independence. |

### Model A Lumped (ERA5, 1950-2023)

| File or folder | Purpose |
|---|---|
| .ipynb_checkpoints | Jupyter autosave checkpoints. |
| FILE_GUIDE.md | This file inventory and Jupyter instructions. |
| Model_A_Lumped_ERA5_1950_2023.ipynb | Extended ERA5 monitoring/results notebook. Run All refreshes saved outputs without training. |
| README.md | Experiment overview and usage. |
| Run_Full_Training_and_Test.ipynb | Run the full model from Jupyter. ACTION=inspect is read-only; ACTION=new_run creates a separate timestamped training/test folder. |
| __pycache__ | Automatic Python import cache. |
| completion.json | Whole-pipeline completion marker, coverage and test summaries. |
| coverage_audit.csv | Source checks and eligible train/validation/test windows. Full-record version also lists observed coverage dates/counts. |
| evaluate_full_record.py | Run 2015–latest tests, save predictions for every reserved ID and explicitly mark missing observations/windows. |
| experiment_config.json | Input names, source paths, date cutoffs, quality rules and exact donor/holdout IDs. |
| extended_record_verification.json | Verify overlap with prior ERA5 cache and newly included earlier/recent observations. |
| fast_dates.py | Decode and align daily NetCDF time bounds, including streamflow observations. |
| final_verification.json | Final checks of all reserved IDs, prediction-file counts, script/notebook syntax and preserved prior model hashes. |
| full_record_coverage_summary.json | Actual full-record date bounds and counts with usable validation/test data. |
| inspect_full_record.py | Render extended-period status, input mapping, coverage, losses and both tests. |
| pipeline.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| pipeline_status.json | Current full-pipeline stage, including temporal evaluation or failures. |
| preparation.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| prepare_full_record.py | Prepare 1950–early-2023 ERA5 plus full discharge observations; completed per-basin caches can be reused after interruption. |
| prepared | One NPZ per catchment: dates, x[days,7] meteorology, y[days] observed discharge in mm/day, static[5]. NaN means missing. Full-record caches also contain audit metadata. |
| prior_experiment_hashes.json | Protected Daymet and original ERA5 file hashes, checked after the full-record run. |
| run_full_record.py | Full-record pipeline: checks, train/resume, both tests and preservation verification. |
| run_seed42 | Model checkpoints, fitted preprocessing, loss history and all test outputs for seed 42. |
| shared_polygon_overlap_screen.csv | Donor/holdout pairs sharing polygon IDs; not an exhaustive geometry audit. |
| source_snapshot | Copies of run source/configuration and SHA-256 hashes for reproducibility. |
| split_manifest.csv | One row per basin: identity, country/group/climate proxy, fixed split and eligible window counts. |
| status.json | Preparation/training or unseen-test stage status; does not necessarily mean later temporal testing is finished. |
| stratum_counts.csv | Fixed catchment counts by split, basin group, climate proxy and country. |
| temporal_evaluation.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| train_unseen.py | Fit donor-only preprocessing, train the shared GPU LSTM, select the validation checkpoint and evaluate unseen catchments. |
| training.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| training_basins.txt | Exact 1326 training donor IDs. |
| training_core_verification.json | AST check that training/preprocessing/validation logic stayed unchanged. |
| unseen_test_basins.txt | Exact 100 held-out IDs, excluded from fitting and checkpoint selection. |
| verification.log | Console log for the named preparation, verification, training, evaluation or reporting stage. Old preparation optimization/restart logs are historical. |
| verify_protocol.py | Checks donor-only fitting, imputation, equal-basin sampling, temporal boundaries and unseen-label independence. |

## Inside run_seed42

| File or pattern | Purpose |
|---|---|
| best.pt | Validation-selected checkpoint: weights, optimizer, epoch/history, config hash and Python/NumPy/Torch/CUDA RNG states. Use for final prediction. |
| last.pt | Most recent completed-epoch checkpoint; use to resume unfinished training with original settings. |
| preprocessing.npz | Donor-training means/stds, static imputation medians and loss scales. Reuse for prediction; do not refit on testing. |
| fit_metadata.json | Fitted/excluded basin IDs, fitting-period endpoint and config hash. |
| run_config.json | Frozen run/data settings, software/GPU details and validation policy. |
| window_counts.csv | Eligible training-window counts and fixed validation-window counts per donor. |
| history.json | Losses, validation donor count and elapsed time for each epoch. |
| summary.json | Selected epoch and unseen-test medians. |
| unseen_metrics.csv | Catchment-level unseen scores: n, NSE, KGE, bias. Full-record version adds score-availability status. |
| unseen_group_metrics.csv | Scores by basin group, climate proxy and country. |
| BASIN_predictions.csv | Daily observed/predicted discharge in mm/day. Observations may be NaN even when predictions exist. |
| unseen_metrics_with_overlap.csv / overlap_screen_metrics.csv | Optional short-period shared-polygon screening summaries. |
| test_coverage_audit.csv / evaluation_status.json / unseen_evaluation_summary.json | Full-record test coverage, progress and finite-score counts. |
| source_snapshot | Original Daymet source copies. ERA5 also has top-level snapshots. |

## Temporal test folders
Original models: run_seed42/temporal_test_2005_2009/. Full-record model: run_seed42/temporal_test_2015_latest/.
metrics.csv contains donor scores; group_metrics.csv grouped scores; summary.json aggregate results; BASIN_predictions.csv daily predictions. coverage_audit.csv (original) or test_coverage_audit.csv (full record) lists score availability. results.html, if present, renders tables. verification.txt/evaluation_status.json record checks/progress.

## Resuming
The Jupyter notebook calls the same Python implementation with subprocess and displays its output. Keep the GPU allocation active. If interrupted, resume in the same rerun folder with last.pt, not a new random model. The full-record pipeline supports this automatically. Prepared inputs and checkpoints persist in home storage.

## Interpretation
The same 26 unseen IDs share polygon IDs with donors; this is not a fully independent spatial split. Climate proxy labels are inherited from the original Daymet split. Native ERA5 PET/daylength definitions differ from Daymet. Full-record scores use different testing years from the original 2005–2009 tests, so the scores are not directly comparable as a forcing-source experiment.
