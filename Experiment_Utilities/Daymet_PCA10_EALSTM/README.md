# EA-LSTM — Daymet + PCA10 mean/std, H256, dropout 0.2, adaptive max50

The experiment is saved separately in `Desktop/LSTM_RR_Modelling/Model A Lumped (Daymet + PCA10 EA-LSTM H256 Dropout02 Adaptive50)`.

Keep these two inputs together:

- `Daymet_PCA10_EALSTM_H256_Dropout02_Adaptive50.ipynb`
- `daymet_pca10_discharge_1980_2009.csv`

In a GPU JupyterLab session, select **Model A PyTorch (GPU)** and run the notebook cells in order. NumPy, PyTorch, IPython and a Jupyter kernel are required. The notebook generates all protocol metadata and outputs itself. The H256 LSTM seed-42 benchmark and reference preprocessing are embedded for portable comparison. Original source-file checks are additional protections where those files exist, not a model-data dependency.

The dataset is an independent, SHA-256-verified copy. All 1,326 training-donor and 100 unseen IDs, features, split dates, normalization, loss, sampling, batch sizes and window masks match the completed H256 LSTM. Train 1980–1999; validate 2000–2004; test 2005–2009. One layer with 256 memory cells, final-output dropout 0.2, training batch 128, validation/inference batch 256, 365 days, Adam LR 0.001, halve after three non-improving validation epochs down to 0.00001, early stopping after ten non-improving epochs, maximum 50, seed 42. Strictly lower validation loss is an improvement. Test results never select a checkpoint.

## EA equations and efficient implementation

The static PCA descriptors determine a time-constant input gate. Meteorology and hidden state determine forget, candidate and output gates. With PyTorch's i/f/g/o gate order:

`i = sigmoid(W_static * attributes + b_i)`

`f_t, g_t, o_t = gates(meteorology_t, h_(t-1))`

`c_t = f_t * c_(t-1) + i * g_t`

`h_t = o_t * tanh(c_t)`

This standalone implementation evaluates these equations through cuDNN with permanent structural zeros. Meteorology-to-input-gate, hidden-to-input-gate, and static-to-other-gate weights are fixed at zero. Gradient masks prevent Adam from regrowing them; checkpoint checks enforce zero preservation. Thus the standard LSTM execution engine is used, but the gate connectivity is EA-LSTM. Static descriptors are constant across every window by construction and are verified constant per basin when reading the CSV.

PyTorch's two bias vectors and baseline uniform initialization policy are retained. No additional forget-bias boost or recurrent-identity/orthogonal initialization is introduced. The representation stores 292,097 parameter slots, including 82,688 structural zeros; 209,409 slots are active/unmasked. This is a hidden-size-matched comparison, not an active-parameter-count-matched comparison. CPU double-precision and CUDA full-FP32 forward/gradient tests compare against an independent direct recurrence. TF32 is temporarily disabled for numerical-equivalence tests and the baseline numerical policy is restored before training and benchmarking.

References: https://hess.copernicus.org/articles/23/5089/2019/index.html and https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.modelzoo.ealstm.html . NeuralHydrology is not a runtime dependency.

## Outputs and reuse

All outputs are under `outputs/`. `basin_model_performance.csv` contains all 1,426 basins and Seen/Unseen labels, NSE, KGE, RMSE, MAE, signed bias, flow thresholds and subset errors. `basin_baseline_comparison.csv` and `paired_comparison_summary.csv` compare against the completed H256 LSTM baseline. `model_performance_summary.csv` includes score distributions and negative NSE/KGE counts. `model_report.html` displays the comparisons and training curve. `results_seed42/` contains history, preprocessing, best/last epoch checkpoints, selected epoch, and per-basin daily predictions in `unseen100/` and `donor_temporal/`.

Low-flow errors use observed test Q ≤ Q10 and high-flow errors use Q ≥ Q90, including ties. Thresholds use only finite observed/predicted test pairs, after checkpoint selection, and must match the H256 LSTM's thresholds and dates. These are evaluation strata, not training inputs. Report sample counts, RMSE, MAE, mean bias and volume bias. Percentage bias is undefined when observed subset volume is zero. Missing discharge stays missing, and predictions are not clipped.

Rerunning the same notebook resumes a matching completed-epoch checkpoint, including optimizer, scheduling/stopping counters and RNG state. Partial epochs restart from the last completed epoch. Copy only the two inputs into an empty folder to train from scratch. A file lock prevents concurrent runs in the same output folder. This is a single-seed comparison; improvements do not establish repeatability. Existing geographic-overlap limitations are inherited unchanged.
