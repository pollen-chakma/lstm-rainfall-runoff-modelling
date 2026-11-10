"""Shared basin-conditioned LSTM for the exploratory Daymet pilot."""
from pathlib import Path
import argparse, csv, json, random, time, copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
BASE=Path(__file__).resolve().parent

class ModelA(nn.Module):
    def __init__(self, dynamic_dim, static_dim, hidden=64):
        super().__init__()
        self.lstm=nn.LSTM(dynamic_dim+static_dim,hidden,batch_first=True)
        self.head=nn.Linear(hidden,1)
    def forward(self,x,a):
        z=torch.cat([x,a[:,None,:].expand(-1,x.shape[1],-1)],-1)
        z,_=self.lstm(z)
        return self.head(z[:,-1]).squeeze(-1)

def load():
    c=json.loads((BASE/'pilot_config.json').read_text()); records=[]
    for b in c['basins']:
        with np.load(BASE/'prepared'/(b['basin_id']+'.npz')) as f:
            records.append(dict(id=b['basin_id'],**{k:f[k] for k in f.files}))
    return c,records

def preprocessing(c,rs):
    train=[r['dates']<c['validation_start'] for r in rs]
    xx=np.concatenate([r['x'][m&np.isfinite(r['x']).all(1)] for r,m in zip(rs,train)])
    yy=np.concatenate([r['y'][m&np.isfinite(r['y'])] for r,m in zip(rs,train)])
    aa=np.stack([r['static'] for r in rs])
    p=dict(xmean=xx.mean(0),xstd=np.maximum(xx.std(0),1e-6),ymean=yy.mean(),ystd=max(yy.std(),1e-6),amean=aa.mean(0),astd=np.maximum(aa.std(0),1e-6))
    p['loss_scales']=np.array([max(np.nanstd(r['y'][m])/p['ystd'],.1) for r,m in zip(rs,train)])
    return p

class Windows(Dataset):
    def __init__(self,c,rs,p,split):
        self.c,self.rs,self.p,self.split=c,rs,p,split; self.samples=[]
        self.x=[((r['x']-p['xmean'])/p['xstd']).astype('float32') for r in rs]
        self.a=[((r['static']-p['amean'])/p['astd']).astype('float32') for r in rs]
        for bi,r in enumerate(rs):
            d=r['dates']; good=np.isfinite(r['x']).all(1); bad=np.r_[0,np.cumsum(~good)]
            target=np.arange(c['lookback']-1,len(d)); L=c['lookback']
            keep=(bad[target+1]-bad[target+1-L])==0
            if split=='train': keep &= d[target]<c['validation_start']
            elif split=='validation': keep &= (d[target]>=c['validation_start'])&(d[target]<c['test_start'])
            else: keep &= d[target]>=c['test_start']
            # Test prediction dates are independent of test discharge availability.
            if split!='test': keep &= np.isfinite(r['y'][target])
            self.samples.extend((bi,int(t)) for t in target[keep])
        assert self.samples,split
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        bi,t=self.samples[i]; r=self.rs[bi]; L=self.c['lookback']
        x=self.x[bi][t+1-L:t+1]; a=self.a[bi]
        y=(r['y'][t]-self.p['ymean'])/self.p['ystd'] if self.split!='test' else 0.
        return torch.from_numpy(x),torch.from_numpy(a),torch.tensor(y,dtype=torch.float32),bi,t

def scores(y,q):
    good=np.isfinite(y)&np.isfinite(q); y,q=y[good],q[good]
    if len(y)<2 or y.std()==0: return dict(n=len(y),NSE=np.nan,KGE=np.nan,bias_percent=np.nan)
    nse=1-np.sum((q-y)**2)/np.sum((y-y.mean())**2)
    corr=np.corrcoef(y,q)[0,1] if q.std()>0 else np.nan
    kge=1-np.sqrt((corr-1)**2+(q.std()/y.std()-1)**2+(q.mean()/y.mean()-1)**2) if y.mean()!=0 else np.nan
    return dict(n=len(y),NSE=float(nse),KGE=float(kge),bias_percent=float(100*(q.sum()-y.sum())/y.sum()) if y.sum()!=0 else np.nan)

def run(smoke=False,epochs=20,seed=42,steps=200):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA GPU required for this run'
    device=torch.device('cuda:0')
    print('DEVICE',device,torch.cuda.get_device_name(0),'memory_GiB',torch.cuda.get_device_properties(0).total_memory/1024**3,flush=True)
    c,rs=load(); p=preprocessing(c,rs)
    ds={s:Windows(c,rs,p,s) for s in ['train','validation','test']}
    model=ModelA(len(c['features']),len(c['static'])).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    counts=np.bincount([b for b,t in ds['train'].samples],minlength=len(rs))
    weights=[1/counts[b] for b,t in ds['train'].samples]
    sampler=WeightedRandomSampler(weights,num_samples=(32 if smoke else steps*32),replacement=True)
    loader=DataLoader(ds['train'],batch_size=4 if smoke else 32,sampler=sampler)
    x,a,y,bi,t=next(iter(loader)); pred=model(x.to(device),a.to(device))
    scale=torch.as_tensor(p['loss_scales'][bi.numpy()],dtype=torch.float32,device=device)
    loss=(((pred-y.to(device))/scale)**2).mean(); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    # Fitting statistics cannot change if validation/test targets change.
    changed=[dict(r,y=r['y'].copy()) for r in rs]
    for r in changed: r['y'][r['dates']>=c['validation_start']]=999999.
    p2=preprocessing(c,changed)
    assert all(np.allclose(p[k],p2[k]) for k in p)
    print('SMOKE PASS',device,'loss',float(loss.detach()),'windows',{s:len(d) for s,d in ds.items()},flush=True)
    optimizer.zero_grad()
    if smoke: return
    out=BASE/'runs'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'preprocessing.npz',**p)
    (out/'run_config.json').write_text(json.dumps(dict(device=str(device),gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,data=c,seed=seed,epochs=epochs,steps_per_epoch=steps,hidden=64,batch=32,task='same-day rainfall-runoff simulation; includes target-day weather'),indent=2))
    best=float('inf'); history=[]
    vl=DataLoader(ds['validation'],batch_size=128)
    for epoch in range(epochs):
        model.train(); total=[]
        for x,a,y,bi,t in loader:
            optimizer.zero_grad(); pred=model(x.to(device),a.to(device))
            scale=torch.tensor(p['loss_scales'][bi.numpy()],dtype=torch.float32,device=device)
            loss=(((pred-y.to(device))/scale)**2).mean(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); total.append(float(loss.detach()))
        model.eval(); sums=np.zeros(len(rs)); nums=np.zeros(len(rs))
        with torch.no_grad():
            for x,a,y,bi,t in vl:
                err=(model(x.to(device),a.to(device)).cpu().numpy()-y.numpy())**2/p['loss_scales'][bi.numpy()]**2
                np.add.at(sums,bi.numpy(),err); np.add.at(nums,bi.numpy(),1)
        val=float(np.mean(sums/nums)); history.append(dict(epoch=epoch+1,train_loss=float(np.mean(total)),validation_loss=val))
        if val<best:
            best=val; torch.save(model.state_dict(),out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print('EPOCH',history[-1],flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)); model.eval()
    results=[[] for r in rs]
    with torch.no_grad():
        for x,a,y,bi,ti in DataLoader(ds['test'],batch_size=128):
            q=model(x.to(device),a.to(device)).cpu().numpy()*p['ystd']+p['ymean']
            for b,t,v in zip(bi.numpy(),ti.numpy(),q): results[b].append((int(t),float(v)))
    metrics=[]
    for r,values in zip(rs,results):
        ts=np.array([t for t,v in values]); q=np.array([v for t,v in values]); obs=r['y'][ts]
        with open(out/(r['id']+'_predictions.csv'),'w',newline='') as f:
            w=csv.writer(f); w.writerow(['date','observed_mm_day','predicted_mm_day']); w.writerows(zip(r['dates'][ts],obs,q))
        metrics.append(dict(basin_id=r['id'],**scores(obs,q)))
    with open(out/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    print('DONE',out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--seed',type=int,default=42); p.add_argument('--steps',type=int,default=200); a=p.parse_args(); run(a.smoke,a.epochs,a.seed,a.steps)
