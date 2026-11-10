from pathlib import Path
import json, ast, hashlib, shutil, csv, base64, zlib
import numpy as np

R=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
S=R/'Model A Lumped (Daymet + PCA10)'
B=R/'Model A Lumped (Daymet + PCA10 H256 Dropout02 Adaptive50)'
U=Path(__file__).resolve().parent
B.mkdir(exist_ok=True);(B/'outputs').mkdir(exist_ok=True)
source_nb=S/'Daymet_PCA10_LSTM.ipynb'; nb=json.loads(source_nb.read_text())
code={i:''.join(c['source']) for i,c in enumerate(nb['cells']) if c['cell_type']=='code'}
assert set(code)=={2,3,4,6,8,10,12}
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()
src=S/'daymet_pca10_discharge_1980_2009.csv';dest=B/src.name
if not dest.exists():
    print('COPYING DATASET',flush=True);tmp=dest.with_suffix('.csv.tmp');shutil.copyfile(src,tmp);tmp.replace(dest)
source_hash=digest(src);assert digest(dest)==source_hash
assert not dest.is_symlink() and src.stat().st_ino!=dest.stat().st_ino
print('DATASET COPY VERIFIED',source_hash,flush=True)
config=once(code[2],'OUT=CSV_PATH.parent/RUN_NAME',"OUTPUT_BASE=CSV_PATH.parent/'outputs'\nOUTPUT_BASE.mkdir(exist_ok=True)\nOUT=OUTPUT_BASE/RUN_NAME\nOUT.mkdir(exist_ok=True)")
config=once(config,'epochs=20','epochs=50');config=once(config,'hidden=64','hidden=256')
config+= "\nCFG.update(output_dropout=0.2,lr_factor=0.5,lr_patience=3,min_lr=0.00001,early_stopping_patience=10,improvement_rule='strictly lower validation loss',flow_thresholds='test observed Q10/Q90, inclusive; reporting only')\nimport fcntl\nif '_run_lock' not in globals():\n    _run_lock=(OUT/'run.lock').open('a')\n    fcntl.flock(_run_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
defs=code[8].split('model=MeteorologyLSTM().to(DEVICE)',1)[0]
defs=once(defs,'nn.LSTM(27,64,batch_first=True);self.head=nn.Linear(64,1)','nn.LSTM(27,256,batch_first=True);self.output_dropout=nn.Dropout(0.2);self.head=nn.Linear(256,1)')
defs=once(defs,'self.head(z[:,-1])','self.head(self.output_dropout(z[:,-1]))')
# Checkpoint function is extended with scheduler/early stopping state and all RNG states.
begin=defs.index('def checkpoint(');end=defs.index('def validate(')
defs=defs[:begin]+(U/'training_checkpoint.txt').read_text()+defs[end:]
training=(U/'training_loop.txt').read_text()
evaluation=code[10].replace("model='Daymet + PCA10'","model='Daymet + PCA10 mean/std H256 dropout0.2 adaptive50'").replace('CSV_PATH.parent/','OUTPUT_BASE/')
# Retain exact source formulas, scalers, date masks, sampling and inference logic.
assert ast.dump(ast.parse(evaluation.replace("model='Daymet + PCA10 mean/std H256 dropout0.2 adaptive50'","model='Daymet + PCA10'").replace('OUTPUT_BASE/','CSV_PATH.parent/')))==ast.dump(ast.parse(code[10]))
baseline=list(csv.DictReader((S/'basin_model_performance.csv').open()))
assert len(baseline)==len({x['basin_id'] for x in baseline})==1426
# Only benchmark outputs are embedded. They are referenced after model selection, not used in fitting.
for i,row in enumerate(baseline):
    group='donor_temporal' if row['seen_unseen']=='Seen' else 'unseen100'
    path=S/'results_seed42'/group/(row['basin_id']+'_predictions.csv')
    daily=list(csv.DictReader(path.open()));y=np.array([float(x['observed_mm_day']) for x in daily]);q=np.array([float(x['predicted_mm_day']) for x in daily]);good=np.isfinite(y)&np.isfinite(q);y,q=y[good],q[good]
    row['test_dates_sha256']=hashlib.sha256('\n'.join(x['date'] for x in daily).encode()).hexdigest()
    row['mean_bias_mm_day']=float((q-y).mean()) if len(y) else np.nan
    row['negative_predictions']=int(np.sum(q<0));row['negative_prediction_percent']=float(np.mean(q<0)*100) if len(q) else np.nan
    for label,quant in [('low',.1),('high',.9)]:
        threshold=float(np.quantile(y,quant)) if len(y) else np.nan
        keep=y<=threshold if label=='low' else y>=threshold
        a,c=y[keep],q[keep];e=c-a
        row.update({label+'_threshold_mm_day':threshold,label+'_n':len(a),label+'_RMSE':float(np.sqrt(np.mean(e*e))) if len(a) else np.nan,label+'_MAE':float(np.mean(np.abs(e))) if len(a) else np.nan,label+'_bias_mm_day':float(e.mean()) if len(a) else np.nan,label+'_bias_percent':float(100*e.sum()/a.sum()) if len(a) and a.sum()!=0 else np.nan})
    if (i+1)%300==0:print('BASELINE BENCHMARK READ',i+1,flush=True)
reference=base64.b64encode(zlib.compress(json.dumps(baseline).encode())).decode()
report=(U/'evaluation_report.txt').read_text().replace('BASELINE_REFERENCE_DATA',repr(reference))
preserved={str(source_nb):digest(source_nb),str(src):source_hash,**{str(x):digest(x) for x in [S/'basin_model_performance.csv',S/'results_seed42/best.pt',S/'results_seed42/history.json']}}
with np.load(S/'results_seed42/preprocessing.npz') as z:old_scalers={k:z[k].tolist() for k in z.files}
prep=code[6]+ '\n# Verify the preserved preprocessing against the original saved seed-42 scalers.\n_expected_scalers=json.loads('+repr(json.dumps(old_scalers))+')\nassert set(p)==set(_expected_scalers)\nfor _key,_value in _expected_scalers.items():assert np.allclose(p[_key],np.asarray(_value),rtol=1e-12,atol=1e-12),_key\natomic(OUTPUT_BASE/\'preprocessing_matches_baseline.json\',dict(status=\'passed\',all_scalers_and_loss_scales_match=True,source=\'embedded original seed-42 preprocessing values\'))\n'
prep+='\n# Recreate protocol metadata when only the notebook and CSV are copied to an empty folder.\nif not (OUTPUT_BASE/\'protocol_verification.json\').exists():\n    atomic(OUTPUT_BASE/\'protocol_verification.json\',dict(status=\'passed\',csv_sha256=CSV_HASH,rows=1426*10958,basins=1426,seen=1326,unseen=100,schema_and_split_verified=True,preprocessing_matches_baseline=True,preserved_baseline_sha256=json.loads('+repr(json.dumps(preserved))+')))\n'
intro='''# Daymet + PCA10 mean/std: H256, output dropout 0.2, adaptive training

## Two reusable input files
Keep this notebook and `daymet_pca10_discharge_1980_2009.csv` together. Select **Model A PyTorch (GPU)** and run cells in order. The CSV is the sole model dataset. An immutable seed-42 baseline benchmark table is embedded in the final comparison cell, so the comparison does not need the earlier experiment folder. Outputs are created under `outputs/`. Copy the two inputs into an empty folder to train from scratch; rerunning here resumes matching epoch checkpoints.

## Fixed protocol and requested changes
Original 1,326 training donors and 100 unseen IDs; train 1980–1999, validate 2000–2004, test 2005–2009. Identical seven Daymet forcings plus PC01–PC10 mean and std (27 channels), baseline preprocessing, loss, 365-day windows, training batch 128, validation/inference batch 256, 256 sampled windows per donor/epoch, validation cap 128 per donor, gradient clipping 1.0 and seed 42. One LSTM layer changes from 64 to 256 units. Dropout 0.2 is applied to the last LSTM output before the linear head (not recurrent dropout). Model parameter count is 292,097.

Adam starts at 0.001. After **three consecutive non-improving validation epochs**, halve learning rate, minimum 0.00001. This is implemented explicitly to avoid library patience off-by-one differences. Any strictly lower validation loss resets both improvement counters. An LR reduction resets only its three-epoch counter. Stop after ten consecutive non-improving epochs or fifty completed epochs. Select the strictly lowest validation loss; test scores never select settings/checkpoints. Epoch checkpoints retain model/optimizer, scheduler counters, early-stopping counters, history and Python/NumPy/PyTorch/CUDA RNG states. Float32 arithmetic and original prediction handling are retained; no test-discharge clipping.

## Flow errors: definition fixed before evaluation
After selecting the validation-best checkpoint, derive observed discharge Q10 and Q90 separately for each basin from finite observed/predicted pairs in 2005–2009. Low-flow days have observed Q ≤ Q10; high-flow days have Q ≥ Q90. Ties are included, so subsets can exceed 10%. Report subset sample count, RMSE, MAE, mean bias (mm/day) and volume bias (%). A zero observed subset sum gives undefined percent bias; absolute errors remain available. No valid pairs give missing thresholds/errors. These observed test quantiles are **evaluation-only diagnostics**, not training inputs or fitted scaling. The same definitions and dates are applied to the old seed-42 benchmark. They describe relative within-basin extremes, not absolute drought/flood classes.

## Comparison and limits
Report seen and unseen separately, all basin rows including missing-score cases, paired differences on common scored basins, negative NSE/KGE counts, full daily predictions, history and selected epoch. One seed tests the combined architecture/training change, not its individual causes or repeatability. Earlier geographical overlap limitations are inherited unchanged. The 100 unseen basins have already been inspected in previous experiments. GPU results can vary slightly across hardware/software.
'''
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def cc(s):ast.parse(s);return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))
cells=[md(intro),cc(config),md('## 1. Read and verify the dataset\nExact original schema, basin split, dates and SHA-256.'),cc(code[3]+'\n'+code[4]),md('## 2. Prepare inputs\nOriginal donor-only preprocessing, balanced samples and fixed validation windows.'),cc(prep),md('## 3. Define the model and check GPU memory\nThe following training cell tests the unchanged full training/validation batch sizes before fitting.'),cc(defs),md('## 4. Train and validate\nAdaptive LR, epoch checkpoints and validation-only stopping/selection.'),cc(training),md('## 5. Evaluate both seen and unseen catchments\nThis cell runs only after validation-based model selection.'),cc(evaluation),md('## 6. Verify, measure flow errors and compare with seed-42 baseline\nThe embedded reference table is used only in this reporting cell.'),cc(report)]
notebook=dict(nbformat=4,nbformat_minor=5,metadata=nb['metadata'],cells=cells)
path=B/'Daymet_PCA10_H256_Dropout02_Adaptive50.ipynb'
if path.exists():assert [c['source'] for c in json.loads(path.read_text())['cells'] if c['cell_type']=='code']==[c['source'] for c in cells if c['cell_type']=='code'],'Existing notebook differs; not overwritten'
else:path.write_text(json.dumps(notebook,indent=1))
audit=dict(status='passed',source_notebook=str(source_nb),source_csv=str(src),csv_sha256=source_hash,dataset_is_independent_regular_copy=True,rows=1426*10958,basins=1426,seen=1326,unseen=100,preprocessing_and_sampling_code_unchanged=True,evaluation_core_unchanged=True,split_hash='779bde7ce5db72c8158f26f161865c4987eb4ce6a0ac7469383256b41cc2d844',preserved_baseline_sha256=preserved)
(B/'outputs/protocol_verification.json').write_text(json.dumps(audit,indent=2))
(B/'outputs/baseline_reference.json').write_text(json.dumps(baseline))
print('NEW EXPERIMENT READY',str(path),flush=True)
