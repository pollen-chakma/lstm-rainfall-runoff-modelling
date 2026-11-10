from pathlib import Path
import ast,csv,hashlib,json,time,datetime,collections

ROOT=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
SRC=ROOT/'Model A Lumped (Daymet + PCA10)'
B=ROOT/'Model A Lumped (Daymet + PCA10 Means Only)'
B.mkdir(exist_ok=True)
O=B/'outputs';O.mkdir(exist_ok=True)
CSV=B/'daymet_pca10_means_only_discharge_1980_2009.csv'
NB=B/'Daymet_PCA10_Means_Only_LSTM.ipynb'
BASECSV=SRC/'daymet_pca10_discharge_1980_2009.csv'
BASE_NB=SRC/'Daymet_PCA10_LSTM.ipynb'
means=[f'PC{i:02d}_mean' for i in range(1,11)]
basecols=['basin_id','split','date','prcp','tmax','tmin','srad','vp','dayl','pet','q_obs_mm_day']
meta=['seen_unseen','n_subbasins','single_subbasin_flag','time_split']
header=basecols+means+meta
assert BASECSV.is_file() and BASE_NB.is_file()
base=json.loads(BASE_NB.read_text())
code={i:''.join(c['source']) for i,c in enumerate(base['cells']) if c['cell_type']=='code'}
assert set(code)=={2,3,4,6,8,10,12},list(code)
def replace_once(s,a,b,expected=1):
    assert s.count(a)==expected,(a,s.count(a));return s.replace(a,b)
config=code[2]
config=replace_once(config,"'daymet_pca10_discharge_1980_2009.csv'","'daymet_pca10_means_only_discharge_1980_2009.csv'")
config=replace_once(config,'OUT=CSV_PATH.parent/RUN_NAME',"OUTPUT_BASE=CSV_PATH.parent/'outputs'\nOUTPUT_BASE.mkdir(exist_ok=True)\nOUT=OUTPUT_BASE/RUN_NAME")
tree=ast.parse(config)
found=0
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STATIC_FEATURES' for t in n.targets):
        n.value=ast.List(elts=[ast.Constant(v) for v in means],ctx=ast.Load());found+=1
assert found==1
config=ast.unparse(ast.fix_missing_locations(tree))+'\n'
reader=code[3]+'\n'+code[4]
for a,b in [('[11:31]','[11:21]'),('[11:34]','[11:24]'),('chunk[0][31]','chunk[0][21]'),('chunk[0][32]','chunk[0][22]'),('chunk[0][33]','chunk[0][23]'),('r[34]','r[24]')]:
    reader=replace_once(reader,a,b,expected=2 if a=='[11:34]' else 1)
prep=replace_once(code[6],'(365,27)','(365,17)').replace('27 LSTM input channels.','17 LSTM input channels.')
training=code[8]
for a,b in [('nn.LSTM(27,64','nn.LSTM(17,64'),('x.shape[-1]==27','x.shape[-1]==17'),('model.lstm.input_size==27','model.lstm.input_size==17'),('==23873','==21313')]:
    training=replace_once(training,a,b)
defs,train=training.split('model=MeteorologyLSTM().to(DEVICE)',1)
train='model=MeteorologyLSTM().to(DEVICE)'+train
evaluation=code[10].replace("model='Daymet + PCA10'","model='Daymet + PCA10 means only'").replace('CSV_PATH.parent/','OUTPUT_BASE/')
assert "model='Daymet + PCA10 means only'" in evaluation
intro='''# Model A Lumped: Daymet + PCA10 means only

## Two input files
Keep this notebook and **daymet_pca10_means_only_discharge_1980_2009.csv** together in the same directory. Open the notebook there, select **Model A PyTorch (GPU)**, and run all cells in order. No original CAMELS, PCA, split, attribute, utility or other experiment files are required. Python needs NumPy, PyTorch, IPython and a Jupyter kernel. Outputs are generated under `outputs/`.

## Controlled experiment
Seed **42 only**. Exactly the original **1,326 training-donor basins and 100 unseen basins**. Training **1980–1999**, validation **2000–2004**, testing **2005–2009**, separately for seen and unseen catchments. The seven Daymet forcings are prcp, tmax, tmin, srad, vp, dayl and pet. The only static inputs are **PC01_mean through PC10_mean**. No subbasin standard-deviation descriptors, basin IDs, group labels, subbasin counts or split flags enter the model. Standard deviations calculated for normalization are scaling parameters, not extra input features.

The shared PCA basis and means are unchanged from the earlier PCA10 mean/std experiment; PCA is not refitted. One-layer LSTM, **64 hidden units**, linear discharge head, **365-day input windows**, Adam **0.001**, batch **128**, clipping **1**, **20 epochs**, **256 sampled windows per donor per epoch**, and at most **128 fixed validation windows per donor**. Best validation loss selects the checkpoint. There are **17 input channels and 21,313 parameters**; removing ten inputs necessarily changes the input-weight count while the hidden layer is unchanged. Original loss scaling, missing-data handling, window endpoints and prediction handling are preserved.

## Reuse and results
This notebook performs reading, preparation, training, validation, seen/unseen testing, independent metric verification and reporting. `outputs/basin_model_performance.csv` has one row per basin with Seen/Unseen, NSE, KGE, RMSE, MAE, bias, observation count and status. Daily predictions, checkpoints and preprocessing parameters are retained under `outputs/results_seed42/`. `outputs/model_performance_summary.csv` and `outputs/model_report.html` summarize results. Missing discharge stays missing; unavailable scores are not replaced by zero.

To retrain from scratch, copy these two input files into a new empty directory. Rerunning in the same directory resumes a matching checkpoint. GPU arithmetic can vary slightly across hardware/software despite a fixed seed. Earlier meteorology may supply window context for validation/testing; discharge is never a model input. The original geographical-overlap limitations of the basin split remain unchanged.
'''
final=r'''# Independently verify the exported daily predictions against every saved basin score.
import html
with (OUTPUT_BASE/'basin_model_performance.csv').open() as f:
    performance=list(csv.DictReader(f))
assert len(performance)==len({r['basin_id'] for r in performance})==1426
assert sum(r['seen_unseen']=='Seen' for r in performance)==1326
assert sum(r['seen_unseen']=='Unseen' for r in performance)==100
history=json.loads((OUT/'history.json').read_text())
assert [r['epoch'] for r in history]==list(range(1,21))
chosen=json.loads((OUT/'summary.json').read_text())['best_epoch']
assert chosen==min(history,key=lambda r:r['validation_loss'])['epoch']
verified=0
for row in performance:
    group='donor_temporal' if row['seen_unseen']=='Seen' else 'unseen100'
    with (OUT/group/(row['basin_id']+'_predictions.csv')).open() as f:daily=list(csv.DictReader(f))
    dates=[r['date'] for r in daily]
    assert len(dates)==len(set(dates)) and all('2005-01-01'<=d<='2009-12-31' for d in dates)
    y=np.array([r['observed_mm_day'] for r in daily],dtype='float32')
    q=np.array([r['predicted_mm_day'] for r in daily],dtype='float64')
    assert np.isfinite(q).all()
    good=np.isfinite(y);y=y[good];q=q[good];n=len(y);error=q-y
    values=dict(NSE=np.nan,KGE=np.nan,bias_percent=np.nan,RMSE=float(np.sqrt(np.mean(error**2))) if n else np.nan,MAE=float(np.mean(np.abs(error))) if n else np.nan)
    assert n==int(row['n'])
    if n>=2 and y.std()>0:
        values['NSE']=float(1-np.sum(error**2)/np.sum((y-y.mean())**2))
        correlation=np.corrcoef(y,q)[0,1] if q.std()>0 else np.nan
        values['KGE']=float(1-np.sqrt((correlation-1)**2+(q.std()/y.std()-1)**2+(q.mean()/y.mean()-1)**2)) if y.mean()!=0 else np.nan
        values['bias_percent']=float(100*(q.sum()-y.sum())/y.sum()) if y.sum()!=0 else np.nan
    for metric,value in values.items():
        assert np.allclose(value,float(row[metric]),rtol=2e-5,atol=2e-5,equal_nan=True),(row['basin_id'],metric)
    verified+=1
    if verified%200==0:print('METRICS VERIFIED',verified,flush=True)
summaries=[]
for group in ['Seen','Unseen']:
    selected=[r for r in performance if r['seen_unseen']==group]
    for metric in ['NSE','KGE','RMSE','MAE','bias_percent']:
        a=np.array([float(r[metric]) for r in selected]);a=a[np.isfinite(a)]
        summaries.append(dict(seen_unseen=group,metric=metric,total_basins=len(selected),scored_basins=len(a),mean=float(a.mean()) if len(a) else np.nan,median=float(np.median(a)) if len(a) else np.nan,q1=float(np.quantile(a,.25)) if len(a) else np.nan,q3=float(np.quantile(a,.75)) if len(a) else np.nan,negative_count=int(sum(a<0)) if metric in ['NSE','KGE'] else ''))
save_csv(OUTPUT_BASE/'model_performance_summary.csv',summaries)
def table(rows):
    keys=list(rows[0]);return '<div style="overflow:auto"><table style="border-collapse:collapse"><tr>'+''.join('<th style="padding:7px;border:1px solid #ddd">'+html.escape(k.replace('_',' '))+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td style="padding:7px;border:1px solid #ddd">'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table></div>'
report='<html><head><meta charset="utf-8"></head><body style="font:15px sans-serif;margin:24px"><h1>Daymet + PCA10 means only</h1><p>Seed 42. All 20 epochs completed. Best validation checkpoint: epoch '+str(chosen)+'. Test period: 2005–2009. Seven meteorological forcings and ten PCA means; 17 inputs, 64 hidden units.</p><h2>Model performance</h2>'+table(summaries)+'<h2>First ten basin rows</h2>'+table(performance[:10])+'<p>All 1,426 basin rows are in basin_model_performance.csv, including Seen/Unseen labels and missing-score status. Verification independently recalculated metrics from daily predictions for every basin.</p></body></html>'
(OUTPUT_BASE/'model_report.html').write_text(report)
atomic(OUTPUT_BASE/'results_verification.json',dict(status='passed',basins=verified,all_daily_metrics_recomputed=True,epochs=20,seed=42,best_epoch=chosen,inputs=FEATURES+STATIC_FEATURES,parameters=sum(v.numel() for v in model.parameters())))
atomic(OUTPUT_BASE/'completion.json',dict(status='complete',model='Daymet + PCA10 means only',basins=1426,seen=1326,unseen=100,epochs=20,seed=42,best_epoch=chosen))
display(HTML('<h2>Test performance, 2005–2009</h2>'+table(summaries)))
display(HTML('<h2>First ten basin results</h2>'+table(performance[:10])))
print('COMPLETE: all 20 epochs, both test groups and all-basin metric verification. Only the notebook and CSV were required inputs.')
'''
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))
def cc(s):ast.parse(s);return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))
cells=[md(intro),cc(config),md('## 1. Read the combined dataset\nVerify the column schema, original basin split, daily dates and unchanged metadata.\n'),cc(reader),md('## 2. Prepare training and validation inputs\nFit scalers on donor training data only. Create balanced training samples and fixed validation windows.\n'),cc(prep),md('## 3. Define the LSTM, validation and checkpoint functions\n'),cc(defs),md('## 4. Train and validate\nRun all 20 epochs and select the best validation checkpoint.\n'),cc(train),md('## 5. Test seen and unseen catchments\nExport daily predictions and one performance row for every basin.\n'),cc(evaluation),md('## 6. Independently verify and report performance\nRecalculate every metric from daily predictions, export the summary and display the results.\n'),cc(final)]
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Model A PyTorch (GPU)',language='python',name='model_a_torch'),language_info=dict(name='python',version='3.11')),cells=cells)
if NB.exists():
    old=json.loads(NB.read_text());assert [c['source'] for c in old['cells'] if c['cell_type']=='code']==[c['source'] for c in cells if c['cell_type']=='code'],'Existing notebook differs; stop without overwriting'
else:NB.write_text(json.dumps(notebook,indent=1))
# Structural checks show that training, preprocessing and testing retain the baseline logic.
reverse=training
for a,b in [('nn.LSTM(17,64','nn.LSTM(27,64'),('x.shape[-1]==17','x.shape[-1]==27'),('model.lstm.input_size==17','model.lstm.input_size==27'),('==21313','==23873')]:reverse=reverse.replace(a,b)
assert ast.dump(ast.parse(reverse))==ast.dump(ast.parse(code[8]))
assert ast.dump(ast.parse(prep.replace('(365,17)','(365,27)').replace('17 LSTM input channels.','27 LSTM input channels.')))==ast.dump(ast.parse(code[6]))
reverse_eval=evaluation.replace("model='Daymet + PCA10 means only'","model='Daymet + PCA10'").replace('OUTPUT_BASE/','CSV_PATH.parent/')
assert ast.dump(ast.parse(reverse_eval))==ast.dump(ast.parse(code[10]))
(O/'protocol_equivalence.json').write_text(json.dumps(dict(status='passed',source=str(BASE_NB),source_sha256=hashlib.sha256(BASE_NB.read_bytes()).hexdigest(),training_logic_unchanged=True,preparation_logic_unchanged=True,evaluation_logic_unchanged=True,allowed_changes=['ten PCA mean inputs; no subbasin standard deviations','CSV schema positions','17 input channels and 21313 parameters','output paths and model label','self-contained verification/reporting']),indent=2))
audit=O/'dataset_verification.json'
if CSV.exists():
    assert audit.exists() and json.loads(audit.read_text())['status']=='passed','Existing dataset not verified; do not overwrite automatically'
    print('DATASET ALREADY VERIFIED; preparation skipped',flush=True)
else:
    started=time.time();counts=collections.Counter();splits={};expected=hashlib.sha256();tmp=CSV.with_suffix('.csv.tmp')
    with BASECSV.open(newline='') as f,tmp.open('w',newline='') as g:
        reader0=csv.reader(f);h=next(reader0);assert len(h)==35 and len(h)==len(set(h))
        keep=[h.index(k) for k in header];assert not any('_std' in header[i] for i in range(11,21))
        writer=csv.writer(g);writer.writerow(header)
        for row in reader0:
            assert len(row)==len(h)
            selected=[row[i] for i in keep];writer.writerow(selected);expected.update(('|'.join(selected)+'\n').encode())
            counts[selected[0]]+=1;splits.setdefault(selected[0],selected[1]);assert splits[selected[0]]==selected[1]
            if counts[selected[0]]==10958 and len(counts)%200==0:print('CSV EXPORTED BASINS',len(counts),flush=True)
    assert len(counts)==1426 and set(counts.values())=={10958}
    assert sum(s=='training_donor' for s in splits.values())==1326 and sum(s=='unseen_test' for s in splits.values())==100
    actual=hashlib.sha256();n=0
    with tmp.open(newline='') as f:
        r=csv.reader(f);assert next(r)==header
        for row in r:
            assert len(row)==25;actual.update(('|'.join(row)+'\n').encode());n+=1
            if n%2000000==0:print('CSV VERIFIED ROWS',n,flush=True)
    assert n==1426*10958 and actual.digest()==expected.digest()
    tmp.replace(CSV)
    audit.write_text(json.dumps(dict(status='passed',rows=n,basins=1426,seen=1326,unseen=100,columns=25,static_inputs=means,dynamic_inputs=basecols[3:10],target='q_obs_mm_day',source=str(BASECSV),all_projected_values_preserved=True,projection_sha256=actual.hexdigest(),seconds=round(time.time()-started,2)),indent=2))
print('PREPARATION COMPLETE',str(B),flush=True)
