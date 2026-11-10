from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
import torch
from torch.utils.data import DataLoader
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from train_unseen import ModelA,scores,Windows,load_records,atomic,writecsv

def evaluate_set(c,p,model,device,out,ids,label,batch=256):
 out=Path(out); out.mkdir(exist_ok=True); manifest={r['basin_id']:r for r in csv.DictReader(open(BASE/'split_manifest.csv'))}; metrics=[]; audit=[]; started=time.time(); model.eval()
 for i,b in enumerate(ids):
  rs=load_records([b]); r=rs[0]; ts=[]; qs=[]
  try:ds=Windows(c,rs,p,'unseen')
  except AssertionError as exc:
   if str(exc)!='unseen':raise
   ds=None
  if ds is not None:
   with torch.inference_mode():
    for x,a,y,bi,ti in DataLoader(ds,batch_size=batch,num_workers=0,pin_memory=True):
     q=model(x.to(device),a.to(device)).cpu().numpy()*p['ystd']+p['ymean']; ts.extend(ti.numpy().tolist()); qs.extend(q.tolist())
  ts=np.array(ts,dtype=int); pred=np.array(qs,dtype=float); obs=r['y'][ts]; dates=r['dates'][ts]
  assert np.all(dates>=np.datetime64(c['test_start'])) and np.all(dates<np.datetime64(c['end_exclusive'])) and np.isfinite(pred).all()
  assert len(ts)==len(set(ts.tolist()))
  with open(out/(b+'_predictions.csv'),'w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['date','observed_mm_day','predicted_mm_day']); w.writeheader(); w.writerows(dict(date=str(d),observed_mm_day=float(o),predicted_mm_day=float(q)) for d,o,q in zip(dates,obs,pred))
  n=int(np.isfinite(obs).sum()); status='scored' if n>=2 else 'no_valid_forcing_window' if not len(ts) else 'insufficient_observations'
  score=scores(obs,pred) if n>=2 else dict(n=n,NSE=float('nan'),KGE=float('nan'),bias_percent=float('nan'))
  m=manifest[b]; metrics.append(dict(basin_id=b,basin_group=m['basin_group'],climate_class=m['climate_class'],country=m['country'],status=status,**score))
  valid=np.isfinite(obs); audit.append(dict(basin_id=b,predicted_days=len(ts),observed_test_days=n,status=status,first_scored_date=str(dates[valid][0]) if valid.any() else '',last_scored_date=str(dates[valid][-1]) if valid.any() else ''))
  if (i+1)%25==0 or i+1==len(ids):
   atomic(out/'evaluation_status.json',dict(phase='evaluating',test=label,completed=i+1,total=len(ids),elapsed_seconds=round(time.time()-started))); print('EVALUATED',label,i+1,'/',len(ids),flush=True)
 metricfile='unseen_metrics.csv' if label=='unseen' else 'metrics.csv'; groupfile='unseen_group_metrics.csv' if label=='unseen' else 'group_metrics.csv'
 writecsv(out/metricfile,metrics); writecsv(out/'test_coverage_audit.csv',audit); groups=[]
 for field in ['basin_group','climate_class','country']:
  for category in sorted({m[field] for m in metrics}):
   subset=[m for m in metrics if m[field]==category]; scored=[m for m in subset if m['status']=='scored']; groups.append(dict(grouping=field,category=category,basins=len(subset),scored_basins=len(scored),median_NSE=float(np.nanmedian([m['NSE'] for m in scored])) if scored else float('nan'),median_KGE=float(np.nanmedian([m['KGE'] for m in scored])) if scored else float('nan')))
 writecsv(out/groupfile,groups); scored=[m for m in metrics if m['status']=='scored']; finite_nse=[m['NSE'] for m in scored if np.isfinite(m['NSE'])]; finite_kge=[m['KGE'] for m in scored if np.isfinite(m['KGE'])]
 result=dict(requested_basins=len(ids),scored_basins=len(scored),finite_NSE_basins=len(finite_nse),finite_KGE_basins=len(finite_kge),prediction_files=len(list(out.glob('*_predictions.csv'))),test_start=c['test_start'],requested_end_exclusive=c['end_exclusive'],latest_scored_date=max((a['last_scored_date'] for a in audit),default=''),median_NSE=float(np.median(finite_nse)) if finite_nse else float('nan'),median_KGE=float(np.median(finite_kge)) if finite_kge else float('nan'),negative_NSE_basins=sum(v<0 for v in finite_nse),policy='Frozen validation-selected model and training-donor preprocessing; all valid test windows, no test subsampling; absent observations excluded; catchment membership unchanged.')
 atomic(out/('unseen_evaluation_summary.json' if label=='unseen' else 'summary.json'),result); atomic(out/'evaluation_status.json',dict(phase='complete',**result)); return result

def main():
 c=json.loads((BASE/'experiment_config.json').read_text()); run=BASE/'run_seed42'; checkpoint=run/'best.pt'; before=hashlib.sha256(checkpoint.read_bytes()).hexdigest(); assert torch.cuda.is_available(); torch.set_num_threads(2); device=torch.device('cuda:0')
 ck=torch.load(checkpoint,map_location=device,weights_only=False); model=ModelA(7,5,hidden=64).to(device); model.load_state_dict(ck['model']); p=dict(np.load(run/'preprocessing.npz'))
 result=evaluate_set(c,p,model,device,run/'temporal_test_2015_latest',c['training_basins'],'temporal'); result.update(best_epoch=ck['epoch'],checkpoint_sha256=before)
 assert before==hashlib.sha256(checkpoint.read_bytes()).hexdigest(); atomic(run/'temporal_test_2015_latest/summary.json',result); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
