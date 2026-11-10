# Setup 1: Daymet + PCA10 → discharge
A self-contained regional LSTM experiment. The only data input is daymet_pca10_discharge_1980_2009.csv beside this notebook. All preparation, fitting, checkpoint resumption, evaluation and testing code is included below; no external project script, attribute file, split file or model checkpoint is required for a fresh run.

Predictors: seven Daymet meteorological inputs plus the area-weighted mean and standard deviation of PC01–PC10 (20 static predictors). The single-subbasin flag and basin labels are retained only as metadata. Target: observed daily discharge, mm/day.

Original protocol: 1,326 donor basins, 100 unseen basins; train 1980–1999, validate 2000–2004, test 2005–2009. One LSTM layer, hidden size 64, linear discharge head, 365-day window, Adam 0.001, batch 128, 256 sampled windows/basin/epoch, 20 epochs, seed 42, gradient clipping 1. Validation uses at most 128 fixed windows per donor basin; tests use all eligible days. Discharge is never an input. Existing missing-flow days stay missing. The original window convention begins endpoint index 365; retained for parity.

PCA preprocessing was fitted only on donor subbasins. The new LSTM static scaler is fitted once per donor basin. Test and validation observations never fit scalers, weights or PCA. Validation alone selects the best checkpoint. The first validation/test windows may use earlier meteorology as context. This does not expose earlier discharge to the model. Basin IDs held out from fitting may still have geographic overlap with donors, as documented in the PCA audit.

## How to rerun
Select a GPU Python kernel with numpy, torch, pandas, ipykernel, jupyter_client and optionally matplotlib installed. Open from this directory and Run All. Training resumes results_seed42/last.pt only if the configuration and CSV hash match; to train independently change RUN_NAME below. GPU arithmetic may differ slightly across software/hardware despite the fixed seed.


Run Daymet_PCA10_LSTM.ipynb from this folder using the Model A PyTorch (GPU) kernel. Its only data input is daymet_pca10_discharge_1980_2009.csv. Run all cells in order. Existing checkpoints resume completed epochs; use a new RUN_NAME for a fresh run. Outputs: basin_model_performance.csv (all 1426 basins, Seen/Unseen), model_performance_summary.csv, results_seed42/ (scalers, history, checkpoints, daily predictions). Missing observations remain flagged in the performance CSV. Original experiments are unchanged.

Verified completion: 20 epochs; best validation checkpoint: epoch 20. Test period: 2005–2009. Seen: 1317 scored / 1326 total; median NSE 0.664850, KGE 0.640617. Unseen: 100 scored / 100; median NSE 0.583863, KGE 0.551589. Nine seen basins have no test observations. All five metrics independently verified from daily predictions.
Only the notebook and combined CSV are needed to rerun. Preparation and recovery scripts are audit helpers; do not rerun them.
