"""Run on Nibi after inspecting existing jobs. Builds the two self-contained inputs."""
from pathlib import Path
import ast, base64, csv, hashlib, json, os, textwrap, zlib
import numpy as np

ROOT=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
REF=ROOT/'Model A Lumped (Daymet + PCA10 H256 Dropout02 Adaptive50)'
COLAB=os.environ.get('SETUP_A_COLAB')=='1'
DEST=ROOT/('Model A Lumped (Daymet Met Only H256 Dropout02 Adaptive50'+(' Colab)' if COLAB else ')'))
U=Path(__file__).resolve().parent
DEST.mkdir(exist_ok=True);O=DEST/'outputs';O.mkdir(exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
reference_nb=REF/'Daymet_PCA10_H256_Dropout02_Adaptive50.ipynb'
nb=json.loads(reference_nb.read_text());c={i:''.join(x['source']) for i,x in enumerate(nb['cells']) if x['cell_type']=='code'}
assert set(c)=={1,3,5,7,9,11,13}
expected=['hidden=256','epochs=50','output_dropout=0.2','lr_patience=3','early_stopping_patience=10']
assert all(x in c[1] for x in expected)
assert 'nn.LSTM(27,256,batch_first=True)' in c[7] and 'nn.Dropout(0.2)' in c[7]
audit_path=O/'construction_audit.json'
protected=[reference_nb,REF/'daymet_pca10_discharge_1980_2009.csv',REF/'outputs/basin_model_performance.csv',REF/'outputs/results_seed42/preprocessing.npz',REF/'outputs/results_seed42/best.pt',REF/'outputs/results_seed42/history.json']
protected_hash={str(p):sha(p) for p in protected}
if audit_path.exists():assert json.loads(audit_path.read_text())['reference_hashes']==protected_hash
source_csv=REF/'daymet_pca10_discharge_1980_2009.csv';dest_csv=DEST/'daymet_met_only_discharge_1980_2009.csv'
header=['basin_id','split','date','prcp','tmax','tmin','srad','vp','dayl','pet','q_obs_mm_day','seen_unseen','time_split']
if not dest_csv.exists():
    temp=dest_csv.with_suffix('.pending.csv');n=0
    with source_csv.open(newline='') as src,temp.open('w',newline='') as out:
        r=csv.reader(src);h=next(r);ix=[h.index(k) for k in header];w=csv.writer(out);w.writerow(header)
        for row in r:w.writerow([row[i] for i in ix]);n+=1
    assert n==1426*10958;temp.replace(dest_csv)
else:
    if not audit_path.exists():
        # Recover the completed CSV projection if a later notebook-construction assertion failed.
        with dest_csv.open(newline='') as f:
            r=csv.reader(f);assert next(r)==header;assert sum(1 for _ in r)==1426*10958
csv_hash=sha(dest_csv)
if audit_path.exists():assert json.loads(audit_path.read_text())['csv_sha256']==csv_hash
with np.load(REF/'outputs/results_seed42/preprocessing.npz') as z:
    shared_scalers={k:z[k].tolist() for k in z.files if not k.startswith('static_')}
benchmark=list(csv.DictReader((REF/'outputs/basin_model_performance.csv').open()))
assert len(benchmark)==1426 and len({r['basin_id'] for r in benchmark})==1426
reference_payload=base64.b64encode(zlib.compress(json.dumps(benchmark).encode())).decode()

config=c[1]
config=once(config,"CSV_PATH=Path.cwd()/'daymet_pca10_discharge_1980_2009.csv'","CSV_PATH=Path.cwd()/'daymet_met_only_discharge_1980_2009.csv'")
if COLAB:
    config=once(config,"CSV_PATH=Path.cwd()/'daymet_met_only_discharge_1980_2009.csv'",'''try:
    import google.colab
    from google.colab import drive
    drive.mount('/content/drive')
    EXPERIMENT_DIR=Path('/content/drive/MyDrive/LSTM_RR_Modelling/Model A Lumped (Daymet Met Only H256 Dropout02 Adaptive50 Colab)')
except ImportError:
    EXPERIMENT_DIR=Path.cwd()
CSV_PATH=EXPERIMENT_DIR/'daymet_met_only_discharge_1980_2009.csv'
''')
start=config.index('STATIC_FEATURES=');end=config.index('\nCFG=',start);config=config[:start]+'STATIC_FEATURES=[]'+config[end:]
config=once(config,"RUN_NAME='results_seed42'","RUN_NAME='results_seed42'")
start=config.index('import fcntl');config=config[:start]+'''import fcntl
SUITE=CSV_PATH.parent/'outputs';SUITE.mkdir(exist_ok=True)
SEEDS=[42,43,44,45,46]
if '_suite_lock' not in globals():
    _suite_lock=(SUITE/'suite.lock').open('a')
    fcntl.flock(_suite_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
SUITE_STARTED=time.monotonic()
MAX_SESSION_SECONDS=int(os.environ.get('SETUP_A_MAX_SECONDS','12600'))
OUTPUT_BASE=SUITE/'seed_42';OUTPUT_BASE.mkdir(exist_ok=True)
OUT=OUTPUT_BASE/'results_seed42';OUT.mkdir(exist_ok=True)
'''
config+='\nEXPECTED_CSV_SHA256='+repr(csv_hash)+'\nEXPECTED_SHARED_SCALERS=json.loads('+repr(json.dumps(shared_scalers))+')\nREFERENCE_HASHES=json.loads('+repr(json.dumps(protected_hash))+')\n'
if COLAB:
    config=config.replace("MAX_SESSION_SECONDS=int(os.environ.get('SETUP_A_MAX_SECONDS','12600'))","MAX_SESSION_SECONDS=int(os.environ.get('SETUP_A_MAX_SECONDS','36000'))")

reader=c[3]
reader=once(reader,"expected=['basin_id','split','date',*FEATURES,'q_obs_mm_day',*STATIC_FEATURES,'seen_unseen','n_subbasins','single_subbasin_flag','time_split']","expected=['basin_id','split','date',*FEATURES,'q_obs_mm_day','seen_unseen','time_split']")
start=reader.index("        static=np.array(chunk[0][11:31]");end=reader.index('        expected_period=',start)
reader=reader[:start]+"        assert all(r[11]==chunk[0][11] for r in chunk),'Basin group changes'\n        seen='Seen' if split=='training_donor' else 'Unseen';assert chunk[0][11]==seen\n"+reader[end:]
reader=once(reader,'r[34]','r[12]')
reader=once(reader,",static=static,n_subbasins=int(chunk[0][32]),single_subbasin_flag=int(chunk[0][33])",'')
reader+='\nassert CSV_HASH==EXPECTED_CSV_SHA256\nassert STATIC_FEATURES==[]\n'
prep=c[5].split('# Verify the preserved preprocessing',1)[0]
prep=once(prep,"dynamic=self.x[bi][t+1-L:t+1];static=np.broadcast_to(self.rs[bi]['static_scaled'],(L,len(STATIC_FEATURES)))\n        inputs=np.concatenate([dynamic,static],axis=1)","inputs=self.x[bi][t+1-L:t+1].copy()")
start=prep.index('static_train=');end=prep.index("train=Windows",start);prep=prep[:start]+prep[end:]
prep=once(prep,'sampler=EqualBasinSampler(train);','sampler=EqualBasinSampler(train,seed=CFG[\'seed\']);')
prep=prep.replace('(365,27)','(365,7)').replace("print('Static scaler fitted once per donor basin; 27 LSTM input channels.')","print('Seven meteorological channels only; no static inputs.')")
prep+='''
assert set(p)==set(EXPECTED_SHARED_SCALERS)
for k,v in EXPECTED_SHARED_SCALERS.items():assert np.allclose(p[k],np.asarray(v),rtol=1e-12,atol=1e-12),k
atomic(SUITE/'preprocessing_verification.json',dict(status='passed',shared_scalers_match_reference=True,static_inputs_absent=True))
'''
defs=once(c[7],'nn.LSTM(27,256,batch_first=True)','nn.LSTM(7,256,batch_first=True)')
defs=once(defs,'x.shape[-1]==27','x.shape[-1]==7')
defs=defs.replace('Daymet + PCA10 mean/std','Daymet meteorology only')
train=c[9].replace('input_size==27','input_size==7').replace('292097','271617')
train=once(train,'        epoch_start=time.time();',"        if time.monotonic()-SUITE_STARTED>MAX_SESSION_SECONDS:\n            raise SessionBudgetPause('Session budget reached; rerun the notebook to resume saved epochs.')\n        epoch_start=time.time();")
evaluation=c[11].replace('Daymet + PCA10 mean/std H256 dropout0.2 adaptive50','Daymet meteorology only H256 dropout0.2 adaptive50')
ref_eval=evaluation.replace('Daymet meteorology only H256 dropout0.2 adaptive50','Daymet + PCA10 mean/std H256 dropout0.2 adaptive50')
assert ast.dump(ast.parse(ref_eval))==ast.dump(ast.parse(c[11]))
train_payload=repr(train);eval_payload=repr(evaluation)
runtime=(U/'suite_runtime.txt').read_text().replace('TRAIN_PAYLOAD',train_payload).replace('EVALUATION_PAYLOAD',eval_payload).replace('REFERENCE_PAYLOAD',repr(reference_payload))
intro='''# Setup A: Daymet meteorology-only H256 LSTM

Keep this notebook and `daymet_met_only_discharge_1980_2009.csv` together. Select Model A PyTorch (GPU), then run cells in order. These are the only required input files; all model, checking and reporting code is included here. Outputs go under `outputs/seed_42` through `seed_46`.

Seven Daymet inputs; no attributes, embeddings or observed-discharge predictors. Exact original 1,326 donor / 100 unseen IDs; train 1980–1999, validate 2000–2004 on donors, test 2005–2009. One ordinary LSTM layer, H256, output dropout0.2, 365-day windows, batch128, 256 sampled windows/donor/epoch, same basin-scaled loss, gradient clipping1, validation/inference batch256 and fixed validation cap128. Adam .001; halve after3 non-improving epochs to minimum .00001; stop after10 non-improving epochs or50; strict validation-best selection. Removing20 static inputs changes parameter count from292097 to271617. Donor-only shared preprocessing is checked against embedded reference scalers. Validation windows are fixed; training sampler seed+epoch matches the reference algorithm. New seeds43–46 have no verified matching H256 reference runs; only seed42 has an embedded controlled comparator.

The sequential five-seed cell saves complete epoch checkpoints and pauses safely at the session budget (default3h30; configurable with SETUP_A_MAX_SECONDS). Run all again to resume. Fully verified finished seeds are skipped after checking their protocol digest. Failures stop the sequence. Five runs may require more than one four-hour allocation. Session pause is operational, not scientific early stopping.

Low/high flow are within-basin observed-test Q10/Q90 (inclusive ties), calculated only after model selection and used only for reporting. Scores and date/threshold definitions match the reference; percent bias is undefined for zero observed subset volume. Missing observations stay missing, predictions are not clipped. Geographic overlap and repeated prior inspection of the test set limit confirmatory interpretation. This is a comparison of descriptors with the same hidden architecture and training rules, not an exact parameter-count match. No ensemble is computed; per-seed summaries and variation across seed-level summaries are explicitly separate.
'''
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def cc(s):ast.parse(s);return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])
cells=[md(intro),cc(config),md('## 1. Read and audit the sole CSV'),cc(reader),md('## 2. Prepare training and validation data'),cc(prep),md('## 3. Define the ordinary LSTM and validation/checkpoint functions'),cc(defs),md('## 4–7. Automatically train, validate, select, test and independently verify seeds42–46\nThen summarize completed seeds and compare only matching reference seeds. Full progress is saved to outputs/suite_status.json.'),cc(runtime)]
out=DEST/'Daymet_MetOnly_H256_Dropout02_Adaptive50.ipynb'
new=dict(nbformat=4,nbformat_minor=5,metadata=nb['metadata'],cells=cells)
if COLAB:
    new['metadata']=dict(kernelspec=dict(name='python3',display_name='Python 3'),language_info=dict(name='python'),accelerator='GPU',colab=dict(name=out.name))
    cells[0]['source']=[''.join(cells[0]['source']).replace('Select Model A PyTorch (GPU), then run cells in order.','In Colab select a T4 GPU, run cells in order, and authorize your own Google Drive when prompted. The notebook reads the CSV from MyDrive/LSTM_RR_Modelling/'+DEST.name+'/ and saves all outputs there. Outside Colab it reads the two inputs from the working directory.').replace('default3h30','Colab default10h').replace('more than one four-hour allocation','more than one Colab runtime')]
if out.exists():assert [x['source'] for x in json.loads(out.read_text())['cells']]==[x['source'] for x in cells],'Do not overwrite a differing notebook'
else:out.write_text(json.dumps(new,indent=1))
audit=dict(status='passed',reference_hashes=protected_hash,csv_sha256=csv_hash,rows=1426*10958,basins=1426,seen=1326,unseen=100,shared_protocol_checked=True,allowed_changes=['remove20 static channels','new input schema','seed loop and operational time-budget pause','outputs and verification/reporting'],evaluation_ast_identical=True)
audit_path.write_text(json.dumps(audit,indent=2))
print(json.dumps(dict(status='ready',notebook=str(out),csv_bytes=dest_csv.stat().st_size,inputs=7,parameters=271617,seeds=[42,43,44,45,46])),flush=True)
