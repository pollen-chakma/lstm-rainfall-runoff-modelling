from pathlib import Path
import sys,csv,json,random,time,hashlib,traceback,argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
BASE=Path(__file__).resolve().parent
PILOT=BASE.parent/'Model_A'
sys.path.insert(0,str(PILOT))
from train_model_a import ModelA,scores

def atomic(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)
def writecsv(path,rows):
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def load_records(ids):
 out=[]
 for i,b in enumerate(ids):
  with np.load(BASE/'prepared'/(b+'.npz')) as z:
   r=dict(id=b,dates=z['dates'].astype('datetime64[D]'),x=z['x'].astype('float32'),y=z['y'].astype('float32'),static=z['static'])
  out.append(r)
  if (i+1)%200==0: print('LOADED',i+1,'training donors',flush=True)
 return out

def fit_preprocessing(c,rs):
 assert {r['id'] for r in rs}==set(c['training_basins'])
 assert not {r['id'] for r in rs}&set(c['unseen_basins'])
 sx=np.zeros(7); sx2=np.zeros(7); nx=0; sy=0.; sy2=0.; ny=0; sigmas=[]
 for r in rs:
  train=r['dates']<np.datetime64(c['validation_start'])
  xx=r['x'][train&np.isfinite(r['x']).all(1)].astype('float64')
  yy=r['y'][train&np.isfinite(r['y'])].astype('float64')
  assert len(xx)>0 and len(yy)>0,r['id']
  sx+=xx.sum(0); sx2+=(xx*xx).sum(0); nx+=len(xx)
  sy+=yy.sum(); sy2+=(yy*yy).sum(); ny+=len(yy); sigmas.append(float(yy.std()))
 aa=np.stack([r['static'] for r in rs]); med=np.nanmedian(aa,axis=0); assert np.isfinite(med).all()
 aa=np.where(np.isfinite(aa),aa,med)
 p=dict(xmean=sx/nx,xstd=np.maximum(np.sqrt(np.maximum(sx2/nx-(sx/nx)**2,0)),1e-6),ymean=sy/ny,ystd=max(np.sqrt(max(sy2/ny-(sy/ny)**2,0)),1e-6),amedian=med,amean=aa.mean(0),astd=np.maximum(aa.std(0),1e-6))
 p['loss_scales']=np.maximum(np.array(sigmas)/p['ystd'],.1)
 return p

class Windows(Dataset):
 def __init__(self,c,rs,p,split,cap=None):
  self.c,self.rs,self.p,self.split=c,rs,p,split; self.indices=[]; self.x=[]; self.a=[]
  for r in rs:
   x=((r['x']-p['xmean'])/p['xstd']).astype('float32'); self.x.append(x)
   a=np.where(np.isfinite(r['static']),r['static'],p['amedian']); self.a.append(((a-p['amean'])/p['astd']).astype('float32'))
   d=r['dates']; target=np.arange(c['lookback']-1,len(d)); bad=np.r_[0,np.cumsum(~np.isfinite(x).all(1))]
   keep=(bad[target+1]-bad[target+1-c['lookback']])==0
   if split=='train': keep &= d[target]<np.datetime64(c['validation_start'])
   elif split=='validation': keep &= (d[target]>=np.datetime64(c['validation_start']))&(d[target]<np.datetime64(c['test_start']))
   elif split=='unseen': keep &= d[target]>=np.datetime64(c['test_start'])
   else: raise ValueError(split)
   if split!='unseen': keep &= np.isfinite(r['y'][target])
   ix=target[keep]
   if cap and len(ix)>cap: ix=ix[np.linspace(0,len(ix)-1,cap).astype(int)]
   self.indices.append(ix.astype('int32'))
  self.lengths=np.array([len(ix) for ix in self.indices]); self.ends=np.cumsum(self.lengths); self.starts=np.r_[0,self.ends[:-1]]
  assert self.ends[-1]>0,split
  if split=='train': assert (self.lengths>0).all(),[r['id'] for r,n in zip(rs,self.lengths) if n==0]
 def __len__(self): return int(self.ends[-1])
 def __getitem__(self,i):
  bi=int(np.searchsorted(self.ends,i,side='right')); t=int(self.indices[bi][i-self.starts[bi]]); L=self.c['lookback']
  y=(self.rs[bi]['y'][t]-self.p['ymean'])/self.p['ystd'] if self.split!='unseen' else 0.
  return torch.from_numpy(self.x[bi][t+1-L:t+1]),torch.from_numpy(self.a[bi]),torch.tensor(y,dtype=torch.float32),bi,t

class EqualBasinSampler(Sampler):
 def __init__(self,ds,per_basin,seed): self.ds,self.per_basin,self.seed,self.epoch=ds,per_basin,seed,0
 def __len__(self): return len(self.ds.rs)*self.per_basin
 def __iter__(self):
  rng=np.random.default_rng(self.seed+self.epoch)
  bs=np.repeat(np.arange(len(self.ds.rs)),self.per_basin); rng.shuffle(bs)
  ix=self.ds.starts[bs]+rng.integers(self.ds.lengths[bs])
  return iter(ix.tolist())

def save_checkpoint(path,model,opt,epoch,best,history,digest):
 state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best_validation=best,history=history,experiment_hash=digest,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
 tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)

def validation(model,loader,p,device,n):
 model.eval(); sums=np.zeros(n); nums=np.zeros(n)
 with torch.no_grad():
  for x,a,y,bi,t in loader:
   pred=model(x.to(device,non_blocking=True),a.to(device,non_blocking=True)).cpu().numpy()
   b=bi.numpy(); err=(pred-y.numpy())**2/p['loss_scales'][b]**2
   sums+=np.bincount(b,weights=err,minlength=n); nums+=np.bincount(b,minlength=n)
 return float(np.mean(sums[nums>0]/nums[nums>0])),int((nums>0).sum())

def evaluate_unseen(c,p,model,device,out,batch):
 from evaluate_full_record import evaluate_set
 atomic(BASE/'status.json',dict(phase='evaluating_unseen',catchments=len(c['unseen_basins']),run_dir=str(out)))
 result=evaluate_set(c,p,model,device,out,c['unseen_basins'],'unseen',batch)
 return dict(unseen_basins=len(c['unseen_basins']),unseen_scored_basins=result['scored_basins'],median_NSE=result['median_NSE'],median_KGE=result['median_KGE'])

def main(args):
 c=json.loads((BASE/'experiment_config.json').read_text()); donors=c['training_basins']; held=c['unseen_basins']
 assert len(donors)==len(set(donors))==1326 and len(held)==len(set(held))==100 and not set(donors)&set(held)
 assert torch.cuda.is_available(),'CUDA GPU required'
 device=torch.device('cuda:0'); torch.set_num_threads(2)
 random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
 torch.backends.cudnn.benchmark=False
 print('GPU',torch.cuda.get_device_name(0),'MEMORY_GiB',torch.cuda.get_device_properties(0).total_memory/1024**3,flush=True)
 atomic(BASE/'status.json',dict(phase='loading_training_donors',training_basins=1326,unseen_basins=100))
 rs=load_records(donors); p=fit_preprocessing(c,rs)
 train=Windows(c,rs,p,'train'); val=Windows(c,rs,p,'validation',cap=args.validation_windows)
 model=ModelA(7,5,hidden=64).to(device); opt=torch.optim.Adam(model.parameters(),lr=.001)
 sampler=EqualBasinSampler(train,args.per_basin,args.seed)
 loader=DataLoader(train,batch_size=args.batch,sampler=sampler,num_workers=0,pin_memory=True)
 x,a,y,bi,t=next(iter(loader)); pred=model(x[:4].to(device),a[:4].to(device)); scale=torch.tensor(p['loss_scales'][bi[:4].numpy()],dtype=torch.float32,device=device)
 loss=(((pred-y[:4].to(device))/scale)**2).mean(); loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
 opt.zero_grad(); torch.cuda.synchronize()
 print('SMOKE PASS CUDA; donors',len(rs),'train windows',len(train),'fixed validation windows',len(val),'unseen excluded from fitting',len(held),flush=True)
 if args.smoke: return
 out=BASE/('run_seed'+str(args.seed))
 if out.exists() and not args.resume: raise FileExistsError('Run exists; use --resume with its last.pt or choose another seed')
 out.mkdir(exist_ok=True)
 protocol=dict(data=c,seed=args.seed,epochs=args.epochs,batch=args.batch,windows_per_basin_per_epoch=args.per_basin,validation_windows_per_basin=args.validation_windows,hidden_units=64,device=str(device),gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,numpy_version=np.__version__,validation_policy='Fixed evenly spaced subset of up to 128 valid days per training donor; unseen catchments never used for checkpoint selection',task='same-day simulation, target-day meteorology included')
 digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
 atomic(out/'run_config.json',protocol); np.savez(out/'preprocessing.npz',**p)
 atomic(out/'fit_metadata.json',dict(fitted_basin_ids=donors,excluded_unseen_basin_ids=held,fit_period_end_exclusive=c['validation_start'],static_imputation='Training-donor median only',experiment_hash=digest))
 writecsv(out/'window_counts.csv',[dict(basin_id=r['id'],train_windows=int(n),fixed_validation_windows=int(v)) for r,n,v in zip(rs,train.lengths,val.lengths)])
 best=float('inf'); history=[]; first_epoch=0
 if args.resume:
  ck=torch.load(args.resume,map_location=device,weights_only=False); assert ck['experiment_hash']==digest
  model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); first_epoch=ck['epoch']; best=ck['best_validation']; history=ck['history']
  random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'].cpu()); torch.cuda.set_rng_state_all(ck['cuda_rng'])
 val_loader=DataLoader(val,batch_size=args.batch*2,num_workers=0,pin_memory=True)
 started=time.time()
 for epoch in range(first_epoch,args.epochs):
  model.train(); sampler.epoch=epoch; total=0.; count=0
  for step,(x,a,y,bi,t) in enumerate(loader,1):
   opt.zero_grad(set_to_none=True); pred=model(x.to(device,non_blocking=True),a.to(device,non_blocking=True)); target=y.to(device,non_blocking=True)
   scale=torch.tensor(p['loss_scales'][bi.numpy()],dtype=torch.float32,device=device)
   loss=(((pred-target)/scale)**2).mean(); assert torch.isfinite(loss),'Nonfinite training loss'
   loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=float(loss.detach())*len(y); count+=len(y)
   if step%200==0:
    atomic(BASE/'status.json',dict(phase='training',epoch=epoch+1,total_epochs=args.epochs,batch=step,total_batches=len(loader),run_dir=str(out),training_basins=1326,unseen_basins=100))
    print('PROGRESS epoch',epoch+1,'batch',step,'/',len(loader),flush=True)
  v,nval=validation(model,val_loader,p,device,len(rs))
  history.append(dict(epoch=epoch+1,training_loss=total/count,validation_loss=v,validation_donors=nval,elapsed_seconds=round(time.time()-started)))
  improved=v<best; best=min(best,v)
  save_checkpoint(out/'last.pt',model,opt,epoch+1,best,history,digest)
  if improved: save_checkpoint(out/'best.pt',model,opt,epoch+1,best,history,digest)
  atomic(out/'history.json',history); print('EPOCH',history[-1],flush=True)
 ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
 # Release donor datasets before loading held-out catchments for final evaluation.
 del loader,val_loader,train,val,rs
 result=evaluate_unseen(c,p,model,device,out,args.batch*2)
 atomic(out/'summary.json',dict(best_epoch=ck['epoch'],**result)); atomic(BASE/'status.json',dict(phase='complete',run_dir=str(out),best_epoch=ck['epoch'],**result))
 print('COMPLETE',out,result,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=20); parser.add_argument('--seed',type=int,default=42); parser.add_argument('--batch',type=int,default=128); parser.add_argument('--per-basin',type=int,default=256); parser.add_argument('--validation-windows',type=int,default=128); parser.add_argument('--resume'); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
 try: main(args)
 except Exception:
  atomic(BASE/'status.json',dict(phase='failed',stage='training',error=traceback.format_exc())); raise
