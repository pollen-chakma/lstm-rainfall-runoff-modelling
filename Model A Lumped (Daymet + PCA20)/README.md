# Model A Lumped — Daymet + PCA20 subbasin summaries

This controlled experiment changes the static representation from PCA10 to PCA20.
Inputs: the same seven Daymet meteorological variables plus 40 static features:
area-weighted means and standard deviations of PC01–PC20. Target: discharge in mm/day.
The shared PCA20 features are already fitted using donor subbasins; PCA is not refitted here.
The first ten components and basin metadata match the PCA10 export.

## Preserved experiment settings
Exactly 1,326 training-donor catchments and 100 unseen catchments, with the original IDs.
Train 1980–1999; validate 2000–2004; test 2005–2009 in both Seen and Unseen groups.
One LSTM layer, hidden size 64, linear discharge head, 365-day input window, Adam 0.001,
batch 128, 256 sampled training windows/basin/epoch, 20 epochs, seed 42, gradient clipping 1.
Validation uses at most 128 fixed windows per donor basin. Loss, sampling, preprocessing,
missing-data policy and window endpoint convention are copied from PCA10.
There are 47 input channels and 28,993 trainable parameters; the additional inputs necessarily
increase the parameter count while the hidden-layer setup stays unchanged.

## Read → prepare → train → validate → test → check performance
Run all cells in order using the Model A PyTorch (GPU) kernel. This notebook reads only
`daymet_pca20_discharge_1980_2009.csv`, placed beside it. NumPy, PyTorch, IPython and
Jupyter kernel support must be installed; no pandas dependency is needed.
The static scaler is fitted once per donor basin; meteorological and flow scalers use only
donor training years. Unseen targets do not fit model/scalers. Validation alone selects
the checkpoint. Earlier forcing can provide validation/test context; discharge is never input.
Geographic overlap between some donor and held-out catchments remains as documented previously.

Existing `results_seed42/last.pt` resumes only if configuration and CSV hashes match.
Use a different RUN_NAME for a new independent training run. GPU arithmetic can vary slightly
across hardware/software despite a fixed seed. The single-subbasin flag is metadata, not an input.

Outputs: basin_model_performance.csv (all 1,426 basins with Seen/Unseen and missing-score status), model_performance_summary.csv, results_seed42/ (scalers, checkpoints, history and daily predictions), completion.json. Earlier experiments remain unchanged. Preparation and verification scripts are audit helpers; rerunning the model requires only the notebook and combined CSV, plus the Python environment.
