from pathlib import Path
import json,csv,hashlib,time,collections,copy
R=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
OLD=R/'Model A Meteorology Only (Daymet)';B=R/'Model A Lumped (Daymet + PCA10)';B.mkdir(exist_ok=True)
PCA=R/'Subbasin_Heterogeneity_PCA/basin_pca10_mean_std_features.csv'
BASE=OLD/'daymet_forcing_discharge_1980_2009.csv'
CSV=B/'daymet_pca10_discharge_1980_2009.csv'
NB=B/'Daymet_PCA10_LSTM.ipynb'
assert not NB.exists() and not CSV.exists(),'New experiment already exists; inspect rather than overwrite.'
pc=[f'PC{i:02d}_{s}' for i in range(1,11) for s in ('mean','std')]
with PCA.open(newline='') as f:pr={r['basin_id']:r for r in csv.DictReader(f)}
assert len(pr)==1426
src=json.loads((OLD/'Daymet_Meteorology_Only_LSTM.ipynb').read_text())
def source(i):return ''.join(src['cells'][i]['source'])
def patch(s,a,b):
 assert a in s,('Missing expected source fragment',a)
 return s.replace(a,b)
c1=source(1)
c1=patch(c1,"daymet_forcing_discharge_1980_2009.csv","daymet_pca10_discharge_1980_2009.csv")
c1=patch(c1,"CFG=dict(","STATIC_FEATURES="+repr(pc)+"\nCFG=dict(")
c1=patch(c1,"static_features=[]","static_features=STATIC_FEATURES")
c1=patch(c1,"print('Static inputs: none')","print('Static inputs:',STATIC_FEATURES)")
c2=source(2)
c2=c2[:c2.index('BASELINE_SCORES=')] if 'BASELINE_SCORES=' in c2 else c2
# Keep only the fixed split hash and simple output helpers, without embedded comparison data.
c2='EXPECTED_SPLIT_HASH="779bde7ce5db72c8158f26f161865c4987eb4ce6a0ac7469383256b41cc2d844"\n'+'''def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def save_csv(path,rows,headers=None):
    headers=headers or list(rows[0])
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)
'''
c4='''from itertools import groupby
records=[];preview=[];expected_dates=np.arange(np.datetime64('1980-01-01'),np.datetime64('2010-01-01'))
started=time.time()
with open(CSV_PATH,newline='') as f:
    reader=csv.reader(f);header=next(reader)
    expected=['basin_id','split','date',*FEATURES,'q_obs_mm_day',*STATIC_FEATURES,'seen_unseen','n_subbasins','single_subbasin_flag','time_split']
    assert header==expected,header
    for basin,group in groupby(reader,key=lambda row:row[0]):
        chunk=list(group);splits={r[1] for r in chunk};assert len(splits)==1
        split=next(iter(splits));dates=np.array([r[2] for r in chunk],dtype='datetime64[D]');assert np.array_equal(dates,expected_dates),basin
        data=np.array([r[3:11] for r in chunk],dtype='float32');assert data.shape==(len(dates),8)
        static=np.array(chunk[0][11:31],dtype='float64');assert np.isfinite(static).all()
        assert all(r[11:34]==chunk[0][11:34] for r in chunk),'Static feature or basin metadata changes within basin'
        seen='Seen' if split=='training_donor' else 'Unseen';assert chunk[0][31]==seen
        expected_period=np.where(dates<np.datetime64('2000-01-01'),'train',np.where(dates<np.datetime64('2005-01-01'),'validation','test'))
        assert np.array_equal(np.array([r[34] for r in chunk]),expected_period)
        assert np.all(data[np.isfinite(data[:,-1]),-1]>=0)
        records.append(dict(id=basin,split=split,dates=dates,x=data[:,:7].copy(),y=data[:,7].copy(),static=static,n_subbasins=int(chunk[0][32]),single_subbasin_flag=int(chunk[0][33])))
        if not preview:preview=chunk[:5]
        if len(records)%200==0:print('CSV catchments loaded:',len(records),flush=True)
ids=[r['id'] for r in records];assert len(ids)==len(set(ids))==1426
split_text=''.join(r['id']+'|'+r['split']+chr(10) for r in sorted(records,key=lambda r:r['id']))
assert hashlib.sha256(split_text.encode()).hexdigest()==EXPECTED_SPLIT_HASH,'Catchment split differs from original baseline.'
donors=[r for r in records if r['split']=='training_donor'];held=[r for r in records if r['split']=='unseen_test']
assert len(donors)==1326 and len(held)==100 and not {r['id'] for r in donors}&{r['id'] for r in held}
hash_csv=hashlib.sha256()
with open(CSV_PATH,'rb') as f:
    for part in iter(lambda:f.read(8*1024*1024),b''):hash_csv.update(part)
CSV_HASH=hash_csv.hexdigest();OUT.mkdir(exist_ok=True)
DIGEST=hashlib.sha256(json.dumps(dict(config=CFG,csv_sha256=CSV_HASH),sort_keys=True).encode()).hexdigest()
atomic(OUT/'data_audit.json',dict(csv=str(CSV_PATH),rows=sum(len(r['dates']) for r in records),basins=1426,donors=1326,unseen=100,csv_sha256=CSV_HASH,dynamic_inputs=FEATURES,static_inputs=STATIC_FEATURES,metadata_excluded=['basin_id','split','seen_unseen','n_subbasins','single_subbasin_flag','time_split'],loader_seconds=round(time.time()-started)))
display(HTML('<table border="1">'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in [header]+preview)+'</table>'))
print('CSV is the sole data source. Exact original split verified.')
'''
c6=source(6)
c6=patch(c6,'class Windows(Dataset):',"class Windows(Dataset):")
c6=patch(c6,"p=fit_preprocessing(donors)","p=fit_preprocessing(donors)\nstatic_train=np.stack([r['static'] for r in donors]);p['static_mean']=static_train.mean(0);p['static_std']=np.maximum(static_train.std(0),1e-6)\nassert np.isfinite(p['static_mean']).all() and np.isfinite(p['static_std']).all()\nfor r in records:r['static_scaled']=((r['static']-p['static_mean'])/p['static_std']).astype('float32')\n")
c6=patch(c6,"return torch.from_numpy(self.x[bi][t+1-L:t+1]),torch.tensor(y,dtype=torch.float32),bi,t", "dynamic=self.x[bi][t+1-L:t+1];static=np.broadcast_to(self.rs[bi]['static_scaled'],(L,len(STATIC_FEATURES)))\n        inputs=np.concatenate([dynamic,static],axis=1)\n        return torch.from_numpy(inputs),torch.tensor(y,dtype=torch.float32),bi,t")
c6=patch(c6,'static_features=[]','static_features=STATIC_FEATURES')
c6+= "\nassert train[0][0].shape==(365,27)\nassert not {r['id'] for r in held}&{r['id'] for r in train.rs}\nprint('Static scaler fitted once per donor basin; 27 LSTM input channels.')\n"
c8=source(8)
c8=patch(c8,'nn.LSTM(7,64','nn.LSTM(27,64')
c8=patch(c8,'x.shape[-1]==7','x.shape[-1]==27')
c8=patch(c8,'model.lstm.input_size==7','model.lstm.input_size==27')
c8=patch(c8,'==18753','==23873')
c8=patch(c8,'static_features=[]','static_features=STATIC_FEATURES')
c10=source(10);c10=c10[:c10.index('summary_rows=[]')]
c10+='''all_metrics=[]
for tag,rows in [('Seen',temporal_metrics),('Unseen',unseen_metrics)]:
    for r in rows:
        r=dict(r,seen_unseen=tag,catchment_split='training_donor' if tag=='Seen' else 'unseen_test',test_start='2005-01-01',test_end='2009-12-31',model='Daymet + PCA10')
        r['status']='scored' if r['n']>=2 and np.isfinite(r['NSE']) else ('no_test_observations' if r['n']==0 else 'insufficient_or_constant_observations')
        all_metrics.append(r)
all_metrics.sort(key=lambda r:r['basin_id'])
assert len(all_metrics)==len({r['basin_id'] for r in all_metrics})==1426
assert sum(r['seen_unseen']=='Seen' for r in all_metrics)==1326
assert sum(r['seen_unseen']=='Unseen' for r in all_metrics)==100
save_csv(CSV_PATH.parent/'basin_model_performance.csv',all_metrics,['basin_id','seen_unseen','catchment_split','model','test_start','test_end','n','NSE','KGE','bias_percent','status'])
summary_rows=[]
for tag in ('Seen','Unseen'):
    rows=[r for r in all_metrics if r['seen_unseen']==tag];scored=[r for r in rows if np.isfinite(r['NSE'])]
    summary_rows.append(dict(seen_unseen=tag,total_basins=len(rows),scored_basins=len(scored),median_NSE=float(np.median([r['NSE'] for r in scored])),median_KGE=float(np.nanmedian([r['KGE'] for r in scored])),median_bias_percent=float(np.nanmedian([r['bias_percent'] for r in scored]))))
save_csv(CSV_PATH.parent/'model_performance_summary.csv',summary_rows)
atomic(OUT/'summary.json',dict(model='Daymet + PCA10',best_epoch=ck['epoch'],test_start='2005-01-01',test_end='2009-12-31',groups=summary_rows))
atomic(OUT/'status.json',dict(phase='complete',basins=1426,seen=1326,unseen=100))
print('Both test groups complete. Basin-level metrics exported beside this notebook.')
'''
c12='''import pandas as pd
performance=pd.read_csv(CSV_PATH.parent/'basin_model_performance.csv',dtype={'basin_id':str})
display(pd.DataFrame(summary_rows));display(performance.head(20))
assert len(performance)==1426
try:
    import matplotlib.pyplot as plt
    history_frame=pd.DataFrame(history)
    ax=history_frame.plot(x='epoch',y=['training_loss','validation_loss'],title='Training and validation loss');ax.set_ylabel('Basin-normalized squared error');ax.figure.savefig(OUT/'training_validation_loss.png',bbox_inches='tight');plt.show()
    ax=performance.boxplot(column='NSE',by='seen_unseen',showfliers=False);ax.set_title('2005–2009 basin NSE');ax.set_ylabel('NSE');ax.figure.suptitle('');ax.figure.savefig(OUT/'seen_unseen_nse.png',bbox_inches='tight');plt.show()
except ImportError:
    print('Install matplotlib to show optional plots; numeric metrics are already saved.')
atomic(CSV_PATH.parent/'completion.json',dict(phase='complete',basins=1426,seen=1326,unseen=100,best_epoch=ck['epoch'],csv_sha256=CSV_HASH,experiment_hash=DIGEST,performance_file='basin_model_performance.csv',summary=summary_rows))
print('Run All complete. Keep this notebook and the combined CSV together to rerun in a compatible GPU Python environment.')
'''
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):compile(s,'notebook','exec');return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])
intro='''# Setup 1: Daymet + PCA10 → discharge
A self-contained regional LSTM experiment. The only data input is daymet_pca10_discharge_1980_2009.csv beside this notebook. All preparation, fitting, checkpoint resumption, evaluation and testing code is included below; no external project script, attribute file, split file or model checkpoint is required for a fresh run.

Predictors: seven Daymet meteorological inputs plus the area-weighted mean and standard deviation of PC01–PC10 (20 static predictors). The single-subbasin flag and basin labels are retained only as metadata. Target: observed daily discharge, mm/day.

Original protocol: 1,326 donor basins, 100 unseen basins; train 1980–1999, validate 2000–2004, test 2005–2009. One LSTM layer, hidden size 64, linear discharge head, 365-day window, Adam 0.001, batch 128, 256 sampled windows/basin/epoch, 20 epochs, seed 42, gradient clipping 1. Validation uses at most 128 fixed windows per donor basin; tests use all eligible days. Discharge is never an input. Existing missing-flow days stay missing. The original window convention begins endpoint index 365; retained for parity.

PCA preprocessing was fitted only on donor subbasins. The new LSTM static scaler is fitted once per donor basin. Test and validation observations never fit scalers, weights or PCA. Validation alone selects the best checkpoint. The first validation/test windows may use earlier meteorology as context. This does not expose earlier discharge to the model. Basin IDs held out from fitting may still have geographic overlap with donors, as documented in the PCA audit.

## How to rerun
Select a GPU Python kernel with numpy, torch, pandas, ipykernel, jupyter_client and optionally matplotlib installed. Open from this directory and Run All. Training resumes results_seed42/last.pt only if the configuration and CSV hash match; to train independently change RUN_NAME below. GPU arithmetic may differ slightly across software/hardware despite the fixed seed.
'''
cells=[md(intro),md('## 1. Read files and define the experiment'),code(c1),code(c2),code(c4),md('## 2. Prepare data using training basins and years only\nFit forcing/flow scalers on donor training years and static scalers once per donor basin. Create valid meteorology windows and equal-basin training sampling.'),code(c6),md('## 3. Train and evaluate on validation years\nCheckpoint after every epoch, resume safely if interrupted, and select the lowest donor validation loss.'),code(c8),md('## 4. Test the frozen best model on seen and unseen catchments\nPredict daily flow for 2005–2009 in both groups. Observed flow is used only to score predictions. NSE, original KGE and volume bias (%) use finite observed/predicted pairs; missing/constant cases retain explicit status.'),code(c10),md('## 5. Check performance and export\nOne row per basin, including all 1,426 IDs and Seen/Unseen. No-observation basins remain present with missing metrics. Results are saved in this experiment directory.'),code(c12)]
nb=dict(cells=cells,metadata=src['metadata'],nbformat=4,nbformat_minor=5)
NB.write_text(json.dumps(nb,indent=1))
# CSV construction is a one-time source preparation step; the model notebook reads only its result.
start=time.time();counts=collections.Counter();tmp=CSV.with_suffix('.tmp.csv')
with BASE.open(newline='') as f,tmp.open('w',newline='') as g:
    reader=csv.reader(f);oldheader=next(reader);assert oldheader==['basin_id','split','date','prcp','tmax','tmin','srad','vp','dayl','pet','q_obs_mm_day']
    writer=csv.writer(g);writer.writerow(oldheader+pc+['seen_unseen','n_subbasins','single_subbasin_flag','time_split'])
    previous=None;extra=None;total=0
    for line in reader:
        bid,split,date=line[:3]
        if bid!=previous:
            r=pr[bid];assert r['catchment_split']==split
            extra=[r[k] for k in pc]+[r['seen_unseen'],r['n_subbasins'],r['single_subbasin_flag']];previous=bid
        period='train' if date<'2000-01-01' else ('validation' if date<'2005-01-01' else 'test')
        writer.writerow(line+extra+[period]);counts[bid]+=1;total+=1
        if total%1000000==0:print('EXPORTED',total,flush=True)
assert set(counts)==set(pr) and set(counts.values())=={10958} and sum(counts.values())==15626108
assert not CSV.exists();tmp.replace(CSV)
provenance=dict(status='combined_csv_and_notebook_prepared',rows=sum(counts.values()),basins=1426,seen=1326,unseen=100,dynamic_predictors=oldheader[3:10],static_predictors=pc,static_count=20,model_channels=27,pca_source=str(PCA),meteorology_discharge_source=str(BASE),pca_sha256=hashlib.sha256(PCA.read_bytes()).hexdigest(),export_seconds=round(time.time()-start),chronology=dict(train='1980–1999',validation='2000–2004',test='2005–2009'))
(B/'dataset_provenance.json').write_text(json.dumps(provenance,indent=2))
# Adapt the existing cell-by-cell executor; it writes actual outputs to the ipynb after every cell.
executor=(OLD/'execute_notebook.py').read_text().replace('Daymet_Meteorology_Only_LSTM.ipynb','Daymet_PCA10_LSTM.ipynb')
(B/'execute_notebook.py').write_text(executor)
(B/'README.md').write_text(intro+'\n\nGenerated outputs: basin_model_performance.csv (all basins), model_performance_summary.csv, results_seed42/ (scalers, fit scope, window counts, epoch history, checkpoints and separate seen/unseen daily predictions). The combined CSV contains repeated static values so this notebook needs only one data file. Existing experiments remain untouched. See dataset_provenance.json for source paths and PCA hash.\n')
print('PREPARATION COMPLETE',json.dumps(provenance),flush=True)
