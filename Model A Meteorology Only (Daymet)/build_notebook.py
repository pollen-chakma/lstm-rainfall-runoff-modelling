from pathlib import Path
import json,csv,ast,textwrap,base64,zlib
R=Path.home()/'Desktop/LSTM_RR_Modelling'; S=R/'Model A Lumped (Daymet)'; B=R/'Model A Meteorology Only (Daymet)'
old=(R/'Model_A/train_model_a.py').read_text(); tree=ast.parse(old); metric_code=next(ast.get_source_segment(old,n) for n in tree.body if getattr(n,'name','')=='scores')
manifest=list(csv.DictReader(open(S/'split_manifest.csv'))); split_text=''.join(x['basin_id']+'|'+x['split']+'\n' for x in sorted(manifest,key=lambda x:x['basin_id']))
import hashlib
expected_split_hash=hashlib.sha256(split_text.encode()).hexdigest()
baselines={}
for tag,path in [('unseen',S/'run_seed42/unseen_metrics.csv'),('temporal',S/'run_seed42/temporal_test_2005_2009/metrics.csv')]:
 baselines[tag]={r['basin_id']:{k:float(r[k]) for k in ['n','NSE','KGE']} for r in csv.DictReader(open(path))}
encoded=base64.b64encode(zlib.compress(json.dumps(baselines).encode())).decode()
cells=[]
def md(s):cells.append(dict(cell_type='markdown',id='c'+str(len(cells)),metadata={},source=[s]))
def code(s):
 s=textwrap.dedent(s).strip()+chr(10);compile(s,'notebook_cell','exec');cells.append(dict(cell_type='code',id='c'+str(len(cells)),metadata={},source=[s],execution_count=None,outputs=[]))
md('# Model A: Daymet meteorology only\nThis notebook and daymet_forcing_discharge_1980_2009.csv are the only model/data files required. Install NumPy, PyTorch and a working Jupyter kernel. Use a GPU. No imports from other model scripts, NetCDF files, attribute files or prepared NPZ data occur during modelling.\n\nTrain 1326 donors in 1980–1999; validate donors in 2000–2004; test both 100 unseen IDs and donors in 2005–2009. Seven meteorological inputs only. Same 64-unit LSTM, 365-day lookback, optimizer, sampling, loss and selection protocol as the attribute baseline; its five static inputs are removed. Catchment ID/split/date organize records and never enter the network. Discharge is the same mm/day target. Basin area was used in the original target conversion only, not as a predictor.\n\nRun All for the first run. Completed results_seed42 runs reuse the checkpoint and repeat test evaluation without retraining; use a new RUN_NAME for another fresh run. Unfinished last.pt checkpoints resume. Previous experiments are not modified.')
code('''
from pathlib import Path
import csv,json,time,random,hashlib,base64,zlib,math,os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
from IPython.display import display,HTML
CSV_PATH=Path.cwd()/'daymet_forcing_discharge_1980_2009.csv'
RUN_NAME='results_seed42'
OUT=CSV_PATH.parent/RUN_NAME
FEATURES=['prcp','tmax','tmin','srad','vp','dayl','pet']
CFG=dict(seed=42,epochs=20,batch=128,windows_per_basin=256,validation_cap=128,lookback=365,hidden=64,learning_rate=.001,gradient_clip=1.,train_start='1980-01-01',validation_start='2000-01-01',test_start='2005-01-01',end_exclusive='2010-01-01',features=FEATURES,static_features=[])
assert CSV_PATH.is_file(), 'Put the CSV beside this notebook, or edit CSV_PATH.'
assert torch.cuda.is_available(), 'Select a GPU Jupyter session/kernel.'
torch.set_num_threads(2)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark=False
DEVICE=torch.device('cuda:0')
print('GPU:',torch.cuda.get_device_name(0));print('CSV:',CSV_PATH);print('Only model inputs:',FEATURES);print('Static inputs: none')
''')
code('EXPECTED_SPLIT_HASH='+repr(expected_split_hash)+chr(10)+'BASELINE_SCORES=json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+')))' +chr(10)+'''# Embedded baseline scores are for final comparison only, never fitting or checkpoint selection.
def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def save_csv(path,rows,headers=None):
    headers=headers or list(rows[0])
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)
''')
md('## Read the CSV and inspect data\nRows are grouped by catchment and sorted by date. Missing forcings or discharge are represented by nan. Every date is retained. Only complete 365-day meteorological sequences become model windows. Missing targets are excluded from training/validation and from final scoring. No basin ID embeddings or previous-flow inputs are used.')
code('''
from itertools import groupby
records=[];preview=[];expected_dates=np.arange(np.datetime64('1980-01-01'),np.datetime64('2010-01-01'))
started=time.time()
with open(CSV_PATH,newline='') as f:
    reader=csv.reader(f);header=next(reader)
    assert header==['basin_id','split','date',*FEATURES,'q_obs_mm_day'],header
    for basin,group in groupby(reader,key=lambda row:row[0]):
        chunk=list(group);splits={r[1] for r in chunk};assert len(splits)==1
        dates=np.array([r[2] for r in chunk],dtype='datetime64[D]');assert np.array_equal(dates,expected_dates),basin
        data=np.asarray([r[3:] for r in chunk],dtype='float32');assert data.shape==(len(dates),8)
        assert np.all(data[np.isfinite(data[:,-1]),-1]>=0)
        records.append(dict(id=basin,split=next(iter(splits)),dates=dates,x=data[:,:7].copy(),y=data[:,7].copy()))
        if not preview:preview=chunk[:5]
        if len(records)%200==0:print('CSV catchments loaded:',len(records),flush=True)
ids=[r['id'] for r in records];assert len(ids)==len(set(ids))==1426
split_text=''.join(r['id']+'|'+r['split']+chr(10) for r in sorted(records,key=lambda r:r['id']))
assert hashlib.sha256(split_text.encode()).hexdigest()==EXPECTED_SPLIT_HASH,'Catchment split differs from the original baseline.'
donors=[r for r in records if r['split']=='training_donor'];held=[r for r in records if r['split']=='unseen_test']
assert len(donors)==1326 and len(held)==100 and not {r['id'] for r in donors}&{r['id'] for r in held}
hash_csv=hashlib.sha256()
with open(CSV_PATH,'rb') as f:
    for part in iter(lambda:f.read(8*1024*1024),b''):hash_csv.update(part)
CSV_HASH=hash_csv.hexdigest();OUT.mkdir(exist_ok=True)
DIGEST=hashlib.sha256(json.dumps(dict(config=CFG,csv_sha256=CSV_HASH),sort_keys=True).encode()).hexdigest()
atomic(OUT/'data_audit.json',dict(csv=str(CSV_PATH),rows=sum(len(r['dates']) for r in records),basins=1426,donors=1326,unseen=100,csv_sha256=CSV_HASH,input_columns=FEATURES,attribute_inputs=[],loader_seconds=round(time.time()-started)))
display(HTML('<table border="1"><tr>'+''.join('<th>'+x+'</th>' for x in header)+'</tr>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in preview)+'</table>'))
print('Exact original split verified. CSV loading finished.')
''')
md('## Donor-only normalization, windows and equal-basin sampling')
code('''
def fit_preprocessing(rs):
    assert len(rs)==1326 and all(r['split']=='training_donor' for r in rs)
    sx=np.zeros(7);sx2=np.zeros(7);nx=0;sy=0.;sy2=0.;ny=0;sigmas=[]
    for r in rs:
        train=r['dates']<np.datetime64(CFG['validation_start'])
        x=r['x'][train&np.isfinite(r['x']).all(1)].astype('float64');y=r['y'][train&np.isfinite(r['y'])].astype('float64')
        assert len(x)>0 and len(y)>0,r['id']
        sx+=x.sum(0);sx2+=(x*x).sum(0);nx+=len(x);sy+=y.sum();sy2+=(y*y).sum();ny+=len(y);sigmas.append(float(y.std()))
    p=dict(xmean=sx/nx,xstd=np.maximum(np.sqrt(np.maximum(sx2/nx-(sx/nx)**2,0)),1e-6),ymean=sy/ny,ystd=max(np.sqrt(max(sy2/ny-(sy/ny)**2,0)),1e-6))
    p['loss_scales']=np.maximum(np.array(sigmas)/p['ystd'],.1);return p
class Windows(Dataset):
    def __init__(self,rs,p,split,cap=None):
        self.rs,self.p,self.split=rs,p,split;self.x=[];self.indices=[]
        for r in rs:
            x=((r['x']-p['xmean'])/p['xstd']).astype('float32');self.x.append(x)
            d=r['dates'];L=CFG['lookback'];t=np.arange(L-1,len(d));bad=np.r_[0,np.cumsum(~np.isfinite(x).all(1))];keep=(bad[t+1]-bad[t+1-L])==0
            if split=='train':keep &= d[t]<np.datetime64(CFG['validation_start'])
            elif split=='validation':keep &= (d[t]>=np.datetime64(CFG['validation_start']))&(d[t]<np.datetime64(CFG['test_start']))
            elif split=='test':keep &= d[t]>=np.datetime64(CFG['test_start'])
            else:raise ValueError(split)
            if split!='test':keep &= np.isfinite(r['y'][t])
            ix=t[keep]
            if cap and len(ix)>cap:ix=ix[np.linspace(0,len(ix)-1,cap).astype(int)]
            self.indices.append(ix.astype('int32'))
        self.lengths=np.array([len(a) for a in self.indices]);self.ends=np.cumsum(self.lengths);self.starts=np.r_[0,self.ends[:-1]]
        if split=='train':assert (self.lengths>0).all()
    def __len__(self):return int(self.ends[-1])
    def __getitem__(self,i):
        bi=int(np.searchsorted(self.ends,i,side='right'));t=int(self.indices[bi][i-self.starts[bi]]);L=CFG['lookback']
        y=(self.rs[bi]['y'][t]-self.p['ymean'])/self.p['ystd'] if self.split!='test' else 0.
        return torch.from_numpy(self.x[bi][t+1-L:t+1]),torch.tensor(y,dtype=torch.float32),bi,t
class EqualBasinSampler(Sampler):
    def __init__(self,ds,per_basin=256,seed=42):self.ds,self.per_basin,self.seed,self.epoch=ds,per_basin,seed,0
    def __len__(self):return len(self.ds.rs)*self.per_basin
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);bs=np.repeat(np.arange(len(self.ds.rs)),self.per_basin);rng.shuffle(bs)
        ix=self.ds.starts[bs]+rng.integers(self.ds.lengths[bs]);return iter(ix.tolist())
p=fit_preprocessing(donors)
train=Windows(donors,p,'train');val=Windows(donors,p,'validation',cap=128)
sampler=EqualBasinSampler(train);loader=DataLoader(train,batch_size=128,sampler=sampler,num_workers=0,pin_memory=True)
val_loader=DataLoader(val,batch_size=256,num_workers=0,pin_memory=True)
assert len(sampler)==1326*256
# Changing test discharge cannot change test-window dates or model-input tensors.
r=held[0];altered=dict(r,y=np.full_like(r['y'],np.nan));ta=Windows([r],p,'test');tb=Windows([altered],p,'test')
assert np.array_equal(ta.indices[0],tb.indices[0]);assert torch.equal(ta[0][0],tb[0][0]);assert ta[0][1].item()==0.
np.savez(OUT/'preprocessing.npz',**p)
atomic(OUT/'fit_metadata.json',dict(fitted_ids=[r['id'] for r in donors],excluded_ids=[r['id'] for r in held],fit_end_exclusive='2000-01-01',static_features=[],experiment_hash=DIGEST))
save_csv(OUT/'window_counts.csv',[dict(basin_id=r['id'],train_windows=int(n),validation_windows=int(v)) for r,n,v in zip(donors,train.lengths,val.lengths)])
print('Training windows:',len(train),'fixed validation windows:',len(val),'validation donors:',int((val.lengths>0).sum()))
''')
md('## LSTM and training\nThe architecture has one LSTM layer, 64 hidden units and a linear last-step readout. It receives exactly seven meteorological columns. Removing five static inputs reduces the parameter count by 1280; hidden architecture and optimizer/loss are unchanged. Each epoch samples 256 windows per donor. This is same-day simulation, not a lead-time forecast.')
code('''
class MeteorologyLSTM(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(7,64,batch_first=True);self.head=nn.Linear(64,1)
    def forward(self,x):
        assert x.shape[-1]==7
        z,_=self.lstm(x);return self.head(z[:,-1]).squeeze(-1)
def checkpoint(path,model,opt,epoch,best,history):
    state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best_validation=best,history=history,experiment_hash=DIGEST,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
def validate(model):
    model.eval();sums=np.zeros(1326);nums=np.zeros(1326)
    with torch.no_grad():
        for x,y,bi,t in val_loader:
            q=model(x.to(DEVICE,non_blocking=True)).cpu().numpy();b=bi.numpy();err=(q-y.numpy())**2/p['loss_scales'][b]**2
            sums+=np.bincount(b,weights=err,minlength=1326);nums+=np.bincount(b,minlength=1326)
    return float(np.mean(sums[nums>0]/nums[nums>0])),int((nums>0).sum())
model=MeteorologyLSTM().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001)
assert model.lstm.input_size==7 and sum(v.numel() for v in model.parameters())==18753
x,y,bi,t=next(iter(loader));q=model(x[:4].to(DEVICE));scale=torch.tensor(p['loss_scales'][bi[:4].numpy()],dtype=torch.float32,device=DEVICE)
loss=(((q-y[:4].to(DEVICE))/scale)**2).mean();loss.backward();assert torch.isfinite(loss);opt.zero_grad();torch.cuda.synchronize()
first=0;best=float('inf');history=[]
if (OUT/'last.pt').exists():
    ck=torch.load(OUT/'last.pt',map_location="cpu",weights_only=False);assert ck['experiment_hash']==DIGEST,'Config/CSV changed; use another RUN_NAME.'
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['epoch'];best=ck['best_validation'];history=ck['history']
    random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all(ck['cuda_rng'])
started=time.time()
for epoch in range(first,20):
    model.train();sampler.epoch=epoch;total=0.;count=0
    for step,(x,y,bi,t) in enumerate(loader,1):
        opt.zero_grad(set_to_none=True);q=model(x.to(DEVICE,non_blocking=True));target=y.to(DEVICE,non_blocking=True)
        scale=torch.tensor(p['loss_scales'][bi.numpy()],dtype=torch.float32,device=DEVICE);loss=(((q-target)/scale)**2).mean();assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(y);count+=len(y)
        if step%400==0:atomic(OUT/'status.json',dict(phase='training',epoch=epoch+1,batch=step,batches=len(loader)));print('PROGRESS',epoch+1,step,flush=True)
    v,nval=validate(model);history.append(dict(epoch=epoch+1,training_loss=total/count,validation_loss=v,validation_donors=nval,elapsed_seconds=round(time.time()-started)))
    improved=v<best;best=min(best,v);checkpoint(OUT/'last.pt',model,opt,epoch+1,best,history)
    if improved:checkpoint(OUT/'best.pt',model,opt,epoch+1,best,history)
    atomic(OUT/'history.json',history);print('EPOCH',history[-1],flush=True)
ck=torch.load(OUT/'best.pt',map_location="cpu",weights_only=False);model.load_state_dict(ck['model']);model.eval();print('Best epoch:',ck['epoch'])
atomic(OUT/'run_config.json',dict(config=CFG,csv_sha256=CSV_HASH,model_inputs=FEATURES,static_features=[],parameters=sum(v.numel() for v in model.parameters()),torch_version=torch.__version__,numpy_version=np.__version__,gpu=torch.cuda.get_device_name(0)))
''')
md('## Full test evaluation and comparison\nThe best checkpoint is frozen. Both test sets use every valid 2005–2009 forcing window. Targets are attached only for scoring; missing discharge is not used as an input or filled. Embedded attribute-model scores are consulted only after final prediction.')
code(metric_code+chr(10)+'''
def evaluate(rs,tag):
    folder=OUT/tag;folder.mkdir(exist_ok=True);metrics=[]
    for i,r in enumerate(rs):
        ds=Windows([r],p,'test');ts=[];pred=[]
        with torch.no_grad():
            for x,y,bi,t in DataLoader(ds,batch_size=256,num_workers=0,pin_memory=True):
                q=model(x.to(DEVICE,non_blocking=True)).cpu().numpy()*p['ystd']+p['ymean'];ts.extend(t.numpy().tolist());pred.extend(q.tolist())
        ts=np.array(ts,dtype=int);q=np.array(pred);obs=r['y'][ts]
        assert np.isfinite(q).all() and np.all(r['dates'][ts]>=np.datetime64('2005-01-01')) and np.all(r['dates'][ts]<np.datetime64('2010-01-01'))
        save_csv(folder/(r['id']+'_predictions.csv'),[dict(date=str(r['dates'][t]),observed_mm_day=float(o),predicted_mm_day=float(v)) for t,o,v in zip(ts,obs,q)],['date','observed_mm_day','predicted_mm_day'])
        metrics.append(dict(basin_id=r['id'],**scores(obs,q)))
        if (i+1)%100==0 or i+1==len(rs):print('TEST',tag,i+1,'/',len(rs),flush=True);atomic(OUT/'status.json',dict(phase='testing',test=tag,completed=i+1,total=len(rs)))
    save_csv(folder/'metrics.csv',metrics);return metrics
unseen_metrics=evaluate(held,'unseen100');temporal_metrics=evaluate(donors,'donor_temporal')
summary_rows=[];paired=[]
for tag,metrics in [('unseen',unseen_metrics),('temporal',temporal_metrics)]:
    scored=[m for m in metrics if np.isfinite(m['NSE'])];baseline=BASELINE_SCORES[tag]
    for m in metrics:
        if m['basin_id'] in baseline:
            old=baseline[m['basin_id']];assert m['n']==int(old['n']),(tag,m['basin_id'],'Scored days differ from baseline')
            paired.append(dict(test=tag,basin_id=m['basin_id'],scored_days=m['n'],attribute_NSE=old['NSE'],meteorology_only_NSE=m['NSE'],attribute_minus_no_attribute_NSE=old['NSE']-m['NSE'],attribute_KGE=old['KGE'],meteorology_only_KGE=m['KGE']))
    summary_rows.append(dict(test=tag,reserved_catchments=len(metrics),scored_catchments=len(scored),meteorology_only_median_NSE=float(np.nanmedian([m['NSE'] for m in metrics])),meteorology_only_median_KGE=float(np.nanmedian([m['KGE'] for m in metrics])),attribute_model_median_NSE=float(np.nanmedian([m['NSE'] for m in baseline.values()])),attribute_model_median_KGE=float(np.nanmedian([m['KGE'] for m in baseline.values()]))))
save_csv(OUT/'comparison_by_catchment.csv',paired);save_csv(OUT/'comparison_summary.csv',summary_rows)
result=dict(phase='complete',best_epoch=ck['epoch'],epochs=20,training_catchments=1326,unseen_catchments=100,features=FEATURES,attribute_inputs=[],csv_sha256=CSV_HASH,results=summary_rows)
atomic(OUT/'summary.json',result);atomic(OUT/'status.json',result)
print(json.dumps(result,indent=2))
''')
md('## Results and interpretation\nPositive attribute_minus_no_attribute_NSE in comparison_by_catchment.csv means the attribute model performed better. This is a single-seed ablation. The original spatial-overlap limitation remains: 26 holdout IDs share polygon IDs with training donors. Adding static inputs also changes the input parameter count, while keeping the hidden architecture fixed.')
code('''
headers=list(summary_rows[0]);display(HTML('<table border="1" cellpadding="6"><tr>'+''.join('<th>'+k+'</th>' for k in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+(f'{v:.4f}' if isinstance(v,float) else str(v))+'</td>' for v in row.values())+'</tr>' for row in summary_rows)+'</table>'))
display(HTML('<h3>Training history</h3><table border="1"><tr><th>Epoch</th><th>Training loss</th><th>Validation loss</th></tr>'+''.join(f"<tr><td>{h['epoch']}</td><td>{h['training_loss']:.4f}</td><td>{h['validation_loss']:.4f}</td></tr>" for h in history)+'</table>'))
print('Outputs:',OUT)
''')
# Turn escaped prose newlines into normal markdown line breaks.
for c in cells:
 if c['cell_type']=='markdown':c['source']=[s.replace(chr(92)+'n',chr(10)) for s in c['source']]
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Model A PyTorch (GPU)',language='python',name='model_a_torch')),cells=cells)
(B/'Daymet_Meteorology_Only_LSTM.ipynb').write_text(json.dumps(nb))
print('SELF-CONTAINED NOTEBOOK SAVED; ALL CODE CELLS COMPILE')
