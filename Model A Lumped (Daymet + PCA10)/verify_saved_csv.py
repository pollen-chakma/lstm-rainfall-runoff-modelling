from pathlib import Path
import csv,json,collections,itertools,hashlib,time
R=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
B=R/'Model A Lumped (Daymet + PCA10)'
BASE=R/'Model A Meteorology Only (Daymet)'/'daymet_forcing_discharge_1980_2009.csv'
PCA=R/'Subbasin_Heterogeneity_PCA'/'basin_pca10_mean_std_features.csv'
CSV=B/'daymet_pca10_discharge_1980_2009.csv'
tmp=CSV.with_suffix('.tmp.csv')
pc=[f'PC{i:02d}_{s}' for i in range(1,11) for s in ('mean','std')]
pr={r['basin_id']:r for r in csv.DictReader(PCA.open())}
counts=collections.Counter(); start=time.time()
with BASE.open() as f,tmp.open() as g:
 a=csv.reader(f); b=csv.reader(g); header=next(a); h=next(b)
 assert h==header+pc+['seen_unseen','n_subbasins','single_subbasin_flag','time_split']
 for x,y in itertools.zip_longest(a,b):
  assert x is not None and y is not None
  bid,split,date=x[:3]; r=pr[bid]
  assert y[:11]==x and split==r['catchment_split']
  assert y[11:34]==[r[k] for k in pc]+[r['seen_unseen'],r['n_subbasins'],r['single_subbasin_flag']]
  assert y[34]==('train' if date<'2000-01-01' else 'validation' if date<'2005-01-01' else 'test')
  counts[bid]+=1
assert set(counts)==set(pr) and set(counts.values())=={10958} and sum(counts.values())==15626108
assert not CSV.exists(); tmp.rename(CSV)
info=dict(status='verified',rows=sum(counts.values()),basins=len(counts),days_per_basin=10958,verification_seconds=time.time()-start)
(B/'dataset_verification.json').write_text(json.dumps(info,indent=2))
print(info,flush=True)
