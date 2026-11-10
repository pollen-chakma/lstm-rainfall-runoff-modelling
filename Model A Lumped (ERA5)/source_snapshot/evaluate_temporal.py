from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
import torch
from torch.utils.data import DataLoader
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from train_unseen import ModelA,scores,Windows,load_records,writecsv,atomic

def main():
 c=json.loads((BASE/'experiment_config.json').read_text()); run=BASE/'run_seed42'; out=run/'temporal_test_2005_2009'; out.mkdir(exist_ok=True)
 ids=c['training_basins']; assert len(ids)==len(set(ids))==1326 and not set(ids)&set(c['unseen_basins'])
 checkpoint=run/'best.pt'; before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 p=dict(np.load(run/'preprocessing.npz')); assert torch.cuda.is_available(); torch.set_num_threads(2); device=torch.device('cuda:0')
 ck=torch.load(checkpoint,map_location=device,weights_only=False); model=ModelA(7,5,hidden=64).to(device); model.load_state_dict(ck['model']); model.eval()
 manifest={r['basin_id']:r for r in csv.DictReader(open(BASE/'split_manifest.csv'))}; metrics=[]; audit=[]; started=time.time()
 for i,b in enumerate(ids):
  rs=load_records([b]); r=rs[0]; ds=Windows(c,rs,p,'unseen'); ts=[]; qs=[]
  with torch.inference_mode():
   for x,a,y,bi,ti in DataLoader(ds,batch_size=256,num_workers=0,pin_memory=True):
    q=model(x.to(device),a.to(device)).cpu().numpy()*p['ystd']+p['ymean']; ts.extend(ti.numpy().tolist()); qs.extend(q.tolist())
  ts=np.array(ts,dtype=int); pred=np.array(qs); obs=r['y'][ts]; dates=r['dates'][ts]
  assert len(ts)==len(set(ts.tolist())) and np.all(dates>=np.datetime64('2005-01-01')) and np.all(dates<np.datetime64('2010-01-01')) and np.isfinite(pred).all()
  writecsv(out/(b+'_predictions.csv'),[dict(date=str(d),observed_mm_day=float(o),predicted_mm_day=float(q)) for d,o,q in zip(dates,obs,pred)])
  n=int(np.isfinite(obs).sum()); m=manifest[b]; audit.append(dict(basin_id=b,predicted_days=len(ts),observed_test_days=n,status='scored' if n>=2 else 'insufficient_observations'))
  if n>=2: metrics.append(dict(basin_id=b,basin_group=m['basin_group'],climate_class=m['climate_class'],country=m['country'],**scores(obs,pred)))
  if (i+1)%50==0 or i+1==len(ids):
   atomic(out/'status.json',dict(phase='evaluating',completed=i+1,total=len(ids),elapsed_seconds=round(time.time()-started))); print('EVALUATED',i+1,'/',len(ids),flush=True)
 writecsv(out/'metrics.csv',metrics); writecsv(out/'coverage_audit.csv',audit)
 groups=[]
 for field in ['basin_group','climate_class','country']:
  for category in sorted({m[field] for m in metrics}):
   subset=[m for m in metrics if m[field]==category]; groups.append(dict(grouping=field,category=category,basins=len(subset),median_NSE=float(np.nanmedian([m['NSE'] for m in subset])),median_KGE=float(np.nanmedian([m['KGE'] for m in subset]))))
 writecsv(out/'group_metrics.csv',groups)
 assert before==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 summary=dict(phase='complete',requested_basins=len(ids),scored_basins=len(metrics),prediction_files=len(list(out.glob('*_predictions.csv'))),best_epoch=ck['epoch'],test_start='2005-01-01',test_end='2009-12-31',median_NSE=float(np.nanmedian([m['NSE'] for m in metrics])),median_KGE=float(np.nanmedian([m['KGE'] for m in metrics])),negative_NSE_basins=sum(m['NSE']<0 for m in metrics),checkpoint_sha256=before,policy='Frozen validation-selected checkpoint and saved training-only preprocessing. Full valid forcing windows, no test subsampling. No retraining or checkpoint selection. Antecedent meteorology may precede 2005; all scored target dates are in 2005-2009.')
 atomic(out/'summary.json',summary); atomic(out/'status.json',summary); print(json.dumps(summary,indent=2),flush=True)
 from html import escape
 def table(rows):
  keys=list(rows[0]); return '<table border="1" cellpadding="5"><tr>'+''.join('<th>'+escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
 html='<h2>Temporal test: trained catchments, 2005–2009</h2>'+table([summary])+'<h3>By group</h3>'+table(groups)+'<h3>Catchment metrics</h3>'+table(metrics)
 (out/'results.html').write_text(html)
 nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3.11',language='python',name='python3')),cells=[dict(cell_type='markdown',metadata={},source=['# Model A: temporal test on 1,326 training catchments\n','Frozen best checkpoint; test target dates 2005–2009. Training 1980–1999; validation 2000–2004. This notebook displays saved evaluation results.']),dict(cell_type='code',metadata={},execution_count=1,source=["from pathlib import Path\nfrom IPython.display import HTML, display\ndisplay(HTML((Path.home()/'Desktop/LSTM_RR_Modelling/Model A Lumped (ERA5)/run_seed42/temporal_test_2005_2009/results.html').read_text()))"],outputs=[dict(output_type='display_data',metadata={},data={'text/html':[html],'text/plain':['Temporal test results']})])])
 (BASE/'Model_A_Temporal_Test.ipynb').write_text(json.dumps(nb))
if __name__=='__main__': main()
