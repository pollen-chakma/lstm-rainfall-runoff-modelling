# Model A Meteorology Only (Daymet)

This is the Daymet forcing-only ablation of Model A Lumped (Daymet). Previous experiment folders are preserved. ERA5 forcing-only is a separate subsequent experiment.

## The two files required to train and test
- Daymet_Meteorology_Only_LSTM.ipynb: complete data loading, checks, normalization, LSTM definition, training, checkpoint selection, full tests and comparison. No imports of project scripts or source NetCDF/NPZ/attribute files.
- daymet_forcing_discharge_1980_2009.csv: 15,626,108 daily rows plus header, 1,426 catchments, 1980-01-01 through 2009-12-31, approximately 1.88 GB. Keep beside the notebook.

Install NumPy and a CUDA-enabled PyTorch in a working Jupyter/IPython environment. On Nibi choose Model A PyTorch (GPU); on another machine select an equivalent installed kernel. The current allocation has 24 GB host memory and an H100 10 GB GPU slice.

Open the notebook in its folder and select Run > Run All Cells. CSV_PATH defaults to the current notebook folder. All modelling code is inside the notebook. For a fresh repeat change RUN_NAME from results_seed42 to a new unique folder name before Run All. With the same RUN_NAME, last.pt resumes completed epochs, or skips training if all 20 epochs exist, and re-evaluates tests. Do not run two writers using the same RUN_NAME simultaneously. A changed config/CSV cannot resume an existing checkpoint.

## CSV columns
basin_id: original catchment ID, kept as text.
split: training_donor or unseen_test, embedding the fixed split.
date: daily date YYYY-MM-DD.
prcp: precipitation (mm/day).
tmax, tmin: maximum and minimum temperature (degrees C).
srad: shortwave radiation.
vp: vapour pressure.
dayl: day length.
pet: potential evapotranspiration (mm/day).
q_obs_mm_day: observed discharge converted to runoff depth (mm/day), the target.

Only prcp,tmax,tmin,srad,vp,dayl,pet enter the network. ID, split and date organize data; discharge is a target and never an input. There are no attributes in the CSV or model. Daymet variables retain the original prepared baseline values/units. Missing values are nan. Target unit conversion previously used basin area; area is not supplied as a predictor. Original negative/partial-day discharge exclusions are retained; estimated/ice-affected values follow the baseline policy.

## Fixed experiment
Exactly the same 1,326 donor IDs and 100 unseen IDs as the original Daymet baseline. Catchment order is also preserved. Normalization is fitted only on donor records dated before 2000. Train 1980-1999; validate 2000-2004; test 2005-2009 for unseen and donor catchments separately. Every input sequence contains 365 consecutive valid meteorological days, including the target day. The initial 364 days supply warm-up rather than training targets. No discharge history enters any sequence. Test windows do not depend on availability of test discharge; metrics use finite observations only.

One LSTM layer, 64 hidden units, last-step linear readout. Seven inputs, 18,753 parameters. Attribute baseline had 12 inputs and 20,033 parameters. Adam learning rate 0.001, batch 128, gradient clipping 1, seed 42, 20 epochs. Each epoch samples 256 valid windows per donor (339,456 samples); selection uses a fixed evenly spaced subset of up to 128 validation days per donor. Loss and validation aggregation follow the original basin-scaled squared-error protocol. Best validation checkpoint is frozen before tests.

## Generated results in RUN_NAME
best.pt / last.pt: best and resumable final checkpoints with optimizer and random states.
preprocessing.npz: donor-only normalization and loss scales.
history.json: epoch training/validation losses.
run_config.json, fit_metadata.json, data_audit.json: settings, fitted IDs, CSV hash and data checks.
window_counts.csv: valid training and validation windows per donor.
status.json: current progress or completed summary.
unseen100/: predictions per catchment and metrics.csv for the reserved 100.
donor_temporal/: predictions per catchment and metrics.csv for the 1,326 donor IDs. Catchments without observations have undefined scores.
comparison_by_catchment.csv: paired attribute and forcing-only NSE/KGE; positive attribute_minus_no_attribute_NSE means attributes improved NSE.
comparison_summary.csv / summary.json: aggregate model comparison.

## Preparation/audit files (not required to rerun the notebook)
export_csv.py: original one-time CSV extraction from the earlier prepared Daymet data.
csv_export.log / csv_export_audit.json: export record.
verify_csv.py / csv_verification.log / csv_roundtrip_verification.json: full exact-float32 value comparison against the earlier data.
build_notebook.py: one-time notebook construction; do not rerun over an executed notebook. It read the previous split and embedded previous metric summaries for comparison. Modelling itself does not read the previous experiment.
execute_notebook.py / notebook_execution.log: optional unattended execution of notebook cells in a Jupyter kernel and saved outputs. GUI Run All does not need this helper.

## Interpretation
This is a single-seed controlled ablation, not conclusive evidence across random seeds. Removing five inputs necessarily reduces the input parameter count while hidden architecture stays fixed. This is same-day simulation, not future discharge forecasting. The original split limitation remains: 26 held-out IDs share polygon IDs with donors. Do not describe these as fully independent spatial catchments. Embedded previous-model metrics are used only after testing, never for fitting or checkpoint selection.

## Completed seed-42 results
20 epochs; best validation checkpoint: epoch 15.
{'test': 'unseen', 'reserved_catchments': 100, 'scored_catchments': 100, 'meteorology_only_median_NSE': 0.4850758311378283, 'meteorology_only_median_KGE': 0.3640139994771888, 'attribute_model_median_NSE': 0.5727823461565513, 'attribute_model_median_KGE': 0.5209324124429464}
{'test': 'temporal', 'reserved_catchments': 1326, 'scored_catchments': 1317, 'meteorology_only_median_NSE': 0.47719032407510453, 'meteorology_only_median_KGE': 0.4054382595842253, 'attribute_model_median_NSE': 0.6029214681106941, 'attribute_model_median_KGE': 0.5671651386941885}
final_verification.json confirms successful full execution and checkpoint reuse. All 1,426 prediction files and both model checkpoints stayed identical after rerun. protocol_parity_verification.json confirms exact normalization/window parity with the baseline.
