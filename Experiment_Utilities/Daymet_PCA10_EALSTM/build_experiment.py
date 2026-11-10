from pathlib import Path
import csv,json,ast,hashlib,shutil,base64,zlib
import numpy as np
R=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
S=R/'Model A Lumped (Daymet + PCA10 H256 Dropout02 Adaptive50)'
B=R/'Model A Lumped (Daymet + PCA10 EA-LSTM H256 Dropout02 Adaptive50)'
U=Path(__file__).resolve().parent
benchmark=json.loads((U/'benchmark_results.json').read_text());assert benchmark['status']=='passed'
B.mkdir(exist_ok=True);O=B/'outputs';O.mkdir(exist_ok=True)
nb_source=S/'Daymet_PCA10_H256_Dropout02_Adaptive50.ipynb'
nb=json.loads(nb_source.read_text());code={i:''.join(c['source']) for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
assert set(code)=={1,3,5,7,9,11,13}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
csv_source=S/'daymet_pca10_discharge_1980_2009.csv';csv_dest=B/csv_source.name
if not csv_dest.exists():
 print('COPYING DATASET',flush=True);tmp=csv_dest.with_suffix('.csv.tmp');shutil.copyfile(csv_source,tmp);tmp.replace(csv_dest)
data_hash=sha(csv_source);assert sha(csv_dest)==data_hash
assert not csv_dest.is_symlink() and csv_source.stat().st_ino!=csv_dest.stat().st_ino
print('DATASET VERIFIED',data_hash,flush=True)
protected={str(nb_source):sha(nb_source),str(csv_source):data_hash,**{str(p):sha(p) for p in [S/'outputs/basin_model_performance.csv',S/'outputs/results_seed42/best.pt',S/'outputs/results_seed42/history.json']}}
config=code[1]+"\nCFG.update(architecture='EA-LSTM masked cuDNN',effective_parameters=209409,structural_zero_parameters=82688,static_input_gate_only=True,initialization='same PyTorch uniform policy and two-bias parameterization as H256 LSTM baseline')\n"
prep=code[5].split('# Verify the preserved preprocessing',1)[0]
with np.load(S/'outputs/results_seed42/preprocessing.npz') as z:scalers={k:z[k].tolist() for k in z.files}
prep+='\n_expected_scalers=json.loads('+repr(json.dumps(scalers))+')\nassert set(p)==set(_expected_scalers)\nfor _k,_v in _expected_scalers.items():assert np.allclose(p[_k],np.asarray(_v),rtol=1e-12,atol=1e-12),_k\natomic(OUTPUT_BASE/\'preprocessing_matches_baseline.json\',dict(status=\'passed\',all_scalers_and_loss_scales_match=True,reference=\'H256 mean/std LSTM baseline\'))\n'
prep+='\nif not (OUTPUT_BASE/\'protocol_verification.json\').exists():\n    atomic(OUTPUT_BASE/\'protocol_verification.json\',dict(status=\'passed\',csv_sha256=CSV_HASH,rows=1426*10958,basins=1426,seen=1326,unseen=100,schema_and_split_verified=True,preprocessing_matches_baseline=True,preserved_baseline_sha256=json.loads('+repr(json.dumps(protected))+')))\n'
arch=(U/'architecture.txt').read_text()
defs=code[7][code[7].index('def progress('):]
defs=defs.replace('Daymet + PCA10 mean/std · H256 / dropout 0.2 / adaptive 50','EA-LSTM Daymet + PCA10 mean/std · H256 / dropout 0.2 / adaptive 50')
defs=once(defs,'def checkpoint(path,model,opt,epoch,best,history):\n    state=', 'def checkpoint(path,model,opt,epoch,best,history):\n    model.assert_structural_zeros()\n    state=')
defs=arch+'\n'+defs+'\natomic(OUTPUT_BASE/\'architecture_verification.json\',verify_ea_architecture())\nprint(\'EA-LSTM forward/backward and gate-separation checks passed\',flush=True)\n'
training=code[9]
training=once(training,'assert model.output_dropout.p==0.2 and sum(v.numel() for v in model.parameters())==292097','assert model.output_dropout.p==0.2 and sum(v.numel() for v in model.parameters())==292097\nassert model.effective_parameter_count()==209409\nmodel.assert_structural_zeros()')
training=training.replace("parameters=292097))","parameters=292097,effective_parameters=209409,structural_zero_parameters=82688))")
training=once(training,"model.load_state_dict(state['model']);opt.load_state_dict", "model.load_state_dict(state['model']);model.assert_structural_zeros();opt.load_state_dict")
training=once(training,"model.load_state_dict(ck['model']);model.eval()", "model.load_state_dict(ck['model']);model.assert_structural_zeros();model.eval()")
training=once(training,"parameters=sum(v.numel() for v in model.parameters()),torch_version", "parameters=sum(v.numel() for v in model.parameters()),effective_parameters=model.effective_parameter_count(),architecture='EA-LSTM masked cuDNN',torch_version")
evaluation=code[11].replace("model='Daymet + PCA10 mean/std H256 dropout0.2 adaptive50'","model='EA-LSTM Daymet + PCA10 mean/std H256 dropout0.2 adaptive50'")
assert ast.dump(ast.parse(evaluation.replace("model='EA-LSTM Daymet + PCA10 mean/std H256 dropout0.2 adaptive50'","model='Daymet + PCA10 mean/std H256 dropout0.2 adaptive50'")))==ast.dump(ast.parse(code[11]))
reference=list(csv.DictReader((S/'outputs/basin_model_performance.csv').open()));assert len(reference)==1426
assert all('test_dates_sha256' in r and 'low_RMSE' in r and 'high_RMSE' in r for r in reference)
encoded=base64.b64encode(zlib.compress(json.dumps(reference).encode())).decode()
report=(U/'evaluation_report.txt').read_text().replace('BASELINE_REFERENCE_DATA',repr(encoded))
intro='''# EA-LSTM: Daymet + PCA10 mean/std, H256, dropout 0.2, adaptive max50

## Two reusable input files
Keep this notebook and `daymet_pca10_discharge_1980_2009.csv` together. Select **Model A PyTorch (GPU)** and run cells in order. The CSV is the sole model dataset. The completed H256 LSTM seed-42 benchmark and its preprocessing reference are embedded in this notebook for portable comparison. All outputs and protocol metadata are generated by this notebook under `outputs/`; no other experiment files are required to rerun. Copy these two inputs into an empty folder to train afresh; rerunning in the same folder resumes a matching epoch checkpoint.

## Controlled comparison with the new H256 LSTM
Same 1,326 training donors and 100 unseen IDs; train 1980–1999, validate 2000–2004, test 2005–2009. Same seven Daymet meteorological variables and 20 static PCA descriptors (PC01–PC10 means and standard deviations), donor-only scaling, loss, 365-day windows, physical training batch 128, validation/inference batch 256, 256 sampled windows per donor per epoch, validation cap 128 per donor, Adam, gradient clipping 1.0, seed 42, one recurrent layer with 256 memory cells, output dropout 0.2 and linear head. Same maximum 50 epochs and validation-only checkpoint selection. LR starts at 0.001 and halves after three consecutive non-improving validation epochs, minimum 0.00001. Stop after ten consecutive non-improving epochs. Improvement means strictly lower validation loss. PCA is not refitted. No extra forget-bias boost or orthogonal/recurrent-identity initialization is added.

## What makes this an EA-LSTM
The input gate is **i = sigmoid(W_static × attributes + b_i)** and is constant through the window. Forget, candidate and output gates receive **meteorology and recurrent hidden state only**, with no direct static-input weights. Memory updates use **c_t = f_t × c_(t−1) + i × g_t**; hidden state uses **h_t = o_t × tanh(c_t)**. Static properties influence later gates indirectly through the hidden state, as in the EA equations.

To retain fast cuDNN execution, this notebook evaluates those equations using a dense PyTorch LSTM with permanent structural zeros: meteorology-to-input-gate weights, hidden-to-input-gate weights and static-to-other-gate weights are zero. Gradient masks keep them zero during Adam updates, and checkpoints assert those constraints. Both PyTorch bias vectors are retained; their sum is the effective gate bias. The same uniform initialization policy as the H256 baseline is used. There are **292,097 stored parameter slots, of which 82,688 are structural zeros and 209,409 are active/unmasked trainable slots**. The models have equal hidden size, not equal active parameter count. Direct independent EA recurrence tests verify both forward values and gradients on CPU and CUDA, plus gate separation and zero preservation after optimization.

References: [EA-LSTM paper, Kratzert et al. (2019)](https://hess.copernicus.org/articles/23/5089/2019/index.html); [NeuralHydrology EA-LSTM equations and gate separation](https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.modelzoo.ealstm.html). This is a standalone mathematically equivalent implementation; the NeuralHydrology package is not required.

## Testing, flow errors and limitations
Test results are computed only after validation-best checkpoint selection. Observed test Q10 and Q90 define within-basin low-flow (Q ≤ Q10) and high-flow (Q ≥ Q90) days, inclusive of ties. Quantiles use finite observed/predicted pairs in 2005–2009, only for reporting; they do not enter training or scaling. Thresholds and test dates must match the H256 baseline. Report subset counts, RMSE, MAE, mean bias and volume bias; zero observed subset volume gives undefined percentage bias. Missing discharge remains missing. Negative predictions are not clipped, preserving the baseline's handling.

Save and compare all 1,426 basin rows, seen/unseen separately, NSE/KGE negative counts, low/high-flow errors, daily predictions, training history and best epoch. Epoch checkpoints include optimizer, both stopping counters and Python/NumPy/PyTorch/CUDA RNG states. One seed tests this architectural change under the same training protocol; it does not establish repeatability. Original basin-overlap limitations and repeated inspection of the unseen set remain unchanged. GPU arithmetic may vary slightly across hardware/software.
'''
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def cc(s):ast.parse(s);return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))
cells=[md(intro),cc(config),md('## 1. Read and verify the unchanged CSV'),cc(code[3]),md('## 2. Prepare inputs and verify baseline-equivalent scaling'),cc(prep),md('## 3. Define and independently verify the EA-LSTM'),cc(defs),md('## 4. Train and validate\nFull-batch GPU check, epoch checkpoints, adaptive LR and validation-only selection.'),cc(training),md('## 5. Test seen and unseen catchments'),cc(evaluation),md('## 6. Verify metrics and compare with the H256 LSTM baseline'),cc(report)]
out=dict(nbformat=4,nbformat_minor=5,metadata=nb['metadata'],cells=cells)
dest=B/'Daymet_PCA10_EALSTM_H256_Dropout02_Adaptive50.ipynb'
if dest.exists():assert [x['source'] for x in json.loads(dest.read_text())['cells'] if x['cell_type']=='code']==[x['source'] for x in cells if x['cell_type']=='code'],'Existing notebook differs; stop without overwriting'
else:dest.write_text(json.dumps(out,indent=1))
(O/'protocol_verification.json').write_text(json.dumps(dict(status='passed',source_experiment=str(S),csv_sha256=data_hash,independent_regular_csv_copy=True,rows=1426*10958,basins=1426,seen=1326,unseen=100,reader_and_preprocessing_core_unchanged=True,evaluation_core_unchanged=True,preserved_baseline_sha256=protected,architecture='EA-LSTM via fixed structural masks',stored_parameters=292097,active_parameters=209409,structural_zero_parameters=82688),indent=2))
(O/'benchmark_results.json').write_text(json.dumps(benchmark,indent=2))
(O/'progress.html').write_text('<html><body style="font:16px sans-serif;margin:24px"><h1>EA-LSTM Daymet + PCA10 mean/std</h1><p>Notebook and unchanged CSV prepared. Architecture equivalence and GPU benchmark passed. Awaiting notebook execution.</p></body></html>')
print('EA-LSTM EXPERIMENT READY',str(dest),flush=True)
