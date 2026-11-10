from pathlib import Path
import json,csv,time,hashlib
import numpy as np
R=Path.home()/'Desktop/LSTM_RR_Modelling'; SRC=R/'Model A Lumped (Daymet)'; B=R/'Model A Meteorology Only (Daymet)'; B.mkdir(exist_ok=True)
C=json.loads((SRC/'experiment_config.json').read_text()); M=list(csv.DictReader(open(SRC/'split_manifest.csv')))
assert len(M)==1426 and len(C['training_basins'])==1326 and len(C['unseen_basins'])==100
p=B/'daymet_forcing_discharge_1980_2009.csv'; assert not p.exists(),'CSV exists; inspect before overwriting'; tmp=p.with_suffix('.tmp.csv'); total=0; started=time.time()
with open(tmp,'w',newline='',buffering=1024*1024) as f:
 w=csv.writer(f); w.writerow(['basin_id','split','date',*C['features'],'q_obs_mm_day'])
 for i,m in enumerate(M):
  b=m['basin_id']
  with np.load(SRC/'prepared'/(b+'.npz')) as z:
   x=z['x']; y=z['y']; dates=z['dates']; assert x.shape==(len(dates),7) and len(y)==len(dates)
   w.writerows([b,m['split'],d,*[format(float(v),'.9g') for v in values]] for d,values in zip(dates,np.column_stack([x,y])))
   total+=len(dates)
  if (i+1)%100==0 or i+1==len(M):print('EXPORTED',i+1,'/1426',total,'rows',flush=True)
tmp.replace(p)
meta=dict(csv_name=p.name,rows=total,basins=1426,training_basins=1326,unseen_basins=100,features=C['features'],target='q_obs_mm_day',attributes_in_csv=[],attribute_features_used=[],discharge_note='Same preprocessed mm/day target and quality exclusions as the Daymet attribute baseline. Area was used only in the existing target unit conversion, never as a model input.',elapsed_seconds=round(time.time()-started),bytes=p.stat().st_size)
(B/'csv_export_audit.json').write_text(json.dumps(meta,indent=2)); print('CSV COMPLETE',json.dumps(meta),flush=True)
