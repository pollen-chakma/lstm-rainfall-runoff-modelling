from pathlib import Path
import csv,numpy as np,json,collections
def rows(p):
 with p.open() as f: return list(csv.DictReader(f))
B=Path("/home/pollen26/Desktop/LSTM_RR_Modelling/Model A Lumped (Daymet + PCA10)")
m=rows(B/'basin_model_performance.csv'); assert len(m)==len({r['basin_id'] for r in m})==1426
assert dict(collections.Counter(r['seen_unseen'] for r in m))=={'Seen':1326,'Unseen':100}
ref={r['basin_id']:r for r in rows(B.parent/'Subbasin_Heterogeneity_PCA'/'basin_pca10_mean_std_features.csv')}
for r in m:
 assert r['seen_unseen']==ref[r['basin_id']]['seen_unseen']
 tag='donor_temporal' if r['seen_unseen']=='Seen' else 'unseen100'
 p=rows(B/'results_seed42'/tag/(r['basin_id']+'_predictions.csv'))
 assert len({a['date'] for a in p})==len(p) and all('2005-01-01'<=a['date']<='2009-12-31' for a in p)
 q=np.array([float(a['predicted_mm_day']) for a in p]); y=np.array([float(a['observed_mm_day']) for a in p],dtype=np.float32); assert np.isfinite(q).all()
 good=np.isfinite(y)&np.isfinite(q); y=y[good]; q=q[good]; assert len(y)==int(r['n'])
 if len(y):
  e=q-y; assert np.isclose(np.sqrt(np.mean(e*e)),float(r['RMSE'])) and np.isclose(np.mean(np.abs(e)),float(r['MAE']))
 if len(y)>1 and y.std()>0:
  nse=1-np.sum((q-y)**2)/np.sum((y-y.mean())**2)
  kge=1-np.sqrt((np.corrcoef(y,q)[0,1]-1)**2+(q.std()/y.std()-1)**2+(q.mean()/y.mean()-1)**2)
  bias=100*(q.sum()-y.sum())/y.sum()
  assert np.allclose([nse,kge,bias],[float(r[k]) for k in ('NSE','KGE','bias_percent')],equal_nan=True), (r['basin_id'],[nse,kge,bias],[r[k] for k in ('NSE','KGE','bias_percent')])
nb=json.loads((B/'Daymet_PCA10_LSTM.ipynb').read_text())
assert all(c.get('execution_count') is not None for c in nb['cells'] if c['cell_type']=='code')
assert not any(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[]))
h=json.loads((B/'results_seed42'/'history.json').read_text())
assert len(h)==20
result=dict(status='passed',basins=1426,seen=1326,unseen=100,metrics_recomputed=['NSE','KGE','RMSE','MAE','bias_percent'],epochs=20,basin_status_counts=dict(collections.Counter(r['status'] for r in m)))
(B/'results_verification.json').write_text(json.dumps(result,indent=2)); print(result); print(rows(B/'model_performance_summary.csv'))
