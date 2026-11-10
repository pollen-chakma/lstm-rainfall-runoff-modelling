# LSTM Rainfall-Runoff Modelling

Multi-basin LSTM and EA-LSTM rainfall-runoff modelling experiments, data processing pipelines, subbasin heterogeneity PCA analyses, and evaluation suites developed on the Alliance **Nibi** cluster (`CAMELS-SPAT` / USA & Canada catchments).

---

## Repository Structure (Experiment Folders)

| Folder / File | Description |
|---|---|
| `Model_A/` | Initial CAMELS-SPAT Model A pilot pipeline, H100 training scripts, and audit outputs |
| `Model A Lumped (Daymet)/` | Baseline Lumped Daymet Model A (`Model_A_Unseen100.ipynb`, `Model_A_Temporal_Test.ipynb`, `train_unseen.py`, `evaluate_temporal.py`) |
| `Model A Lumped (ERA5)/` | Lumped ERA5 forcing experiment and paired Daymet vs. ERA5 comparison |
| `Model A Lumped (ERA5, 1950-2023)/` | Extended historical record (1950–2023) ERA5 lumped LSTM experiment |
| `Model A Meteorology Only (Daymet)/` | Meteorology-only (no static catchment attributes) Daymet LSTM experiment |
| `Model A Lumped (Daymet + PCA10)/` | Daymet forcings + Subbasin Heterogeneity PCA10 (mean & std) static features |
| `Model A Lumped (Daymet + PCA10 Means Only)/` | Daymet forcings + PCA10 subbasin means only ablation experiment |
| `Model A Lumped (Daymet + PCA10 H256 Dropout02 Adaptive50)/` | High-capacity (`H=256`, `Dropout=0.2`, adaptive 50-epoch schedule) Daymet + PCA10 LSTM |
| `Model A Lumped (Daymet + PCA10 EA-LSTM H256 Dropout02 Adaptive50)/` | Entity-Aware LSTM (EA-LSTM, `H=256`, `Dropout=0.2`, adaptive 50-epoch schedule) with Daymet + PCA10 |
| `Model A Lumped (Daymet Met Only H256 Dropout02 Adaptive50 Colab)/` | High-capacity (`H=256`) meteorology-only Daymet experiment |
| `Model A Lumped (Daymet + PCA20)/` | Daymet forcings + Subbasin Heterogeneity PCA20 (mean & std) static features |
| `Daymet_PCA10_PCA20_Five_Seed_Comparison/` | 5-seed ensemble & repeatability comparison (`seed_42`–`seed_46`) between PCA10 and PCA20 |
| `Subbasin_Heterogeneity_PCA/` | Subbasin attribute screening, correlation audit, and PCA10/PCA20 feature extraction pipeline |
| `LSTM_Performance_Comparison/` | Cross-experiment NSE/KGE synthesis tables, paired evaluations, and boxplot generators |
| `Poor_Basin_Performance_Analysis/` | Diagnostic analysis of negative NSE/KGE and persistent failure catchments |
| `Lumped_Data_Explorer/` | Interactive catchment hydro-meteorological explorer dashboard (`dashboard.py`, `Open_Dashboard.ipynb`) |
| `Experiment_Utilities/` | Shared experiment builders, two-file input verification scripts, and execution runners |
| `MODEL_A_FILE_GUIDE.md` | Detailed file guide for Model A experiments |
| `requirements.txt` | Python 3.11 + ComputeCanada (`+computecanada`) wheel dependencies exported from Nibi |

---

## Setting Up on Another Alliance HPC Cluster (e.g., Fir)

### 1. Clone the Repository
```bash
cd ~/scratch   # or ~/projects/def-<yourpi>/<username>
git clone https://github.com/pollen-chakma/lstm-rainfall-runoff-modelling.git
cd lstm-rainfall-runoff-modelling
```

### 2. Recreate the Python Virtual Environment
Alliance clusters (Nibi, Fir, Rorqual, Narval) share the same CVMFS software stack:
```bash
module load python/3.11
virtualenv --no-download .venv
source .venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
```

### 3. Transfer Large Forcing Datasets & Checkpoints (Excluded from Git)
Large merged forcing CSVs (`daymet_*_discharge_1980_2009.csv`, `daymet_lumped_daily.csv`, `era5_lumped_daily.csv`), prepared `.npz` caches, and `.pt` checkpoints are excluded via `.gitignore`. Transfer them directly between Nibi and Fir using **Globus** (`alliancecan#nibi` → `alliancecan#fir`) or `rsync`:
```bash
rsync -avzP pollen26@nibi.alliancecan.ca:"/home/pollen26/Desktop/LSTM_RR_Modelling/" ./
```
