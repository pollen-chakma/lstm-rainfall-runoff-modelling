"""Independently verify PCA20 predictions and compare paired basins with PCA10."""
from pathlib import Path
import csv, collections, json, math
import numpy as np
B=Path(__file__).resolve().parent
OLD=B.parent/'Model A Lumped (Daymet + PCA10)'
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
new=read(B/'basin_model_performance.csv');old=read(OLD/'basin_model_performance.csv')
a={r['basin_id']:r for r in old};b={r['basin_id']:r for r in new}
assert len(new)==len(old)==len(a)==len(b)==1426 and set(a)==set(b)
assert collections.Counter(r['seen_unseen'] for r in new)=={'Seen':1326,'Unseen':100}
metrics=['NSE','KGE','RMSE','MAE','bias_percent']
paired=[]
for bid,r in b.items():
    previous=a[bid]
    assert r['seen_unseen']==previous['seen_unseen'] and r['catchment_split']==previous['catchment_split']
    assert r['test_start']==previous['test_start']=='2005-01-01'
    assert r['test_end']==previous['test_end']=='2009-12-31'
    assert r['n']==previous['n']
    tag='donor_temporal' if r['seen_unseen']=='Seen' else 'unseen100'
    p=read(B/'results_seed42'/tag/(bid+'_predictions.csv'))
    assert len(p)==len({z['date'] for z in p}) and all('2005-01-01'<=z['date']<='2009-12-31' for z in p)
    q=np.array([float(z['predicted_mm_day']) for z in p])
    y=np.array([float(z['observed_mm_day']) for z in p],dtype=np.float32)
    assert np.isfinite(q).all()
    valid=np.isfinite(y)&np.isfinite(q);y=y[valid];q=q[valid]
    assert len(y)==int(r['n'])
    if len(y):
        err=q-y
        assert np.isclose(np.sqrt(np.mean(err**2)),float(r['RMSE']))
        assert np.isclose(np.mean(np.abs(err)),float(r['MAE']))
    if len(y)>1 and y.std()>0:
        nse=1-np.sum((q-y)**2)/np.sum((y-y.mean())**2)
        kge=1-np.sqrt((np.corrcoef(y,q)[0,1]-1)**2+(q.std()/y.std()-1)**2+(q.mean()/y.mean()-1)**2)
        bias=100*(q.sum()-y.sum())/y.sum()
        assert np.allclose([nse,kge,bias],[float(r[k]) for k in ('NSE','KGE','bias_percent')],equal_nan=True)
    z=dict(basin_id=bid,seen_unseen=r['seen_unseen'],n_scored_days=int(r['n']),
           PCA10_status=previous['status'],PCA20_status=r['status'])
    for metric in metrics:
        u=float(previous[metric]);v=float(r[metric])
        z['PCA10_'+metric]=u;z['PCA20_'+metric]=v;z['PCA20_minus_PCA10_'+metric]=v-u
    paired.append(z)
n=json.loads((B/'Daymet_PCA20_LSTM.ipynb').read_text())
assert all(c.get('execution_count') is not None for c in n['cells'] if c['cell_type']=='code')
assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
history=json.loads((B/'results_seed42/history.json').read_text());assert len(history)==20
assert [h['epoch'] for h in history]==list(range(1,21))
write(B/'PCA10_vs_PCA20_basin_comparison.csv',sorted(paired,key=lambda z:z['basin_id']))
summary=[]
for group in ['Seen','Unseen']:
    for metric in metrics:
        rows=[z for z in paired if z['seen_unseen']==group and np.isfinite(z['PCA10_'+metric]) and np.isfinite(z['PCA20_'+metric])]
        u=np.array([z['PCA10_'+metric] for z in rows]);v=np.array([z['PCA20_'+metric] for z in rows]);delta=v-u
        improved=delta>0 if metric in ('NSE','KGE') else (np.abs(v)<np.abs(u) if metric=='bias_percent' else delta<0)
        s=dict(seen_unseen=group,metric=metric,n_paired=len(rows),PCA10_mean=float(u.mean()),PCA20_mean=float(v.mean()),
               PCA10_median=float(np.median(u)),PCA20_median=float(np.median(v)),median_paired_change=float(np.median(delta)),
               improved_basins=int(improved.sum()),improved_percent=float(improved.mean()*100),
               PCA10_negative=int(np.sum(u<0)) if metric in ('NSE','KGE') else '',
               PCA20_negative=int(np.sum(v<0)) if metric in ('NSE','KGE') else '')
        summary.append(s)
write(B/'PCA10_vs_PCA20_summary.csv',summary)
verification=dict(status='passed',basins=1426,seen=1326,unseen=100,epochs=20,
                  all_daily_metrics_recomputed=True,same_IDs_years_and_scored_day_counts_as_PCA10=True,
                  code_cells_executed=sum(c['cell_type']=='code' for c in n['cells']),
                  basin_status_counts=dict(collections.Counter(r['status'] for r in new)))
(B/'results_verification.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification),flush=True)
for s in summary:
    if s['metric'] in ('NSE','KGE'):print(json.dumps(s),flush=True)
