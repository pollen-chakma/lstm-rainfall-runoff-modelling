from pathlib import Path
import csv,json,math,hashlib,collections,statistics
R=Path.home()/'Desktop/LSTM_RR_Modelling';B=R/'LSTM_Performance_Comparison';B.mkdir(exist_ok=True)
models=[('Daymet_MetAttr_Test2005_2009','Model A Lumped (Daymet)','run_seed42/temporal_test_2005_2009/metrics.csv','run_seed42/unseen_metrics.csv','1980-1999','2000-2004','2005-2009'),('Daymet_MetOnly_Test2005_2009','Model A Meteorology Only (Daymet)','results_seed42/donor_temporal/metrics.csv','results_seed42/unseen100/metrics.csv','1980-1999','2000-2004','2005-2009'),('ERA5_MetAttr_Test2005_2009','Model A Lumped (ERA5)','run_seed42/temporal_test_2005_2009/metrics.csv','run_seed42/unseen_metrics.csv','1980-1999','2000-2004','2005-2009'),('ERA5_MetAttr_Test2015_2023','Model A Lumped (ERA5, 1950-2023)','run_seed42/temporal_test_2015_latest/metrics.csv','run_seed42/unseen_metrics.csv','1950-2009','2010-2014','2015-early 2023')]
def read(p):
 with open(p,newline='') as f:a=list(csv.DictReader(f))
 assert len(a)==len({r['basin_id'] for r in a}),str(p)
 return {r['basin_id']:r for r in a}
manifest_path=R/'Model A Lumped (Daymet)/split_manifest.csv';master=read(manifest_path);assert len(master)==1426
assert collections.Counter(r['split'] for r in master.values())=={'training_donor':1326,'unseen_test':100}
inputs=[manifest_path];loaded={};stats={}
for prefix,folder,donor,held,train,val,test in models:
 sources={};coverage={}
 for split,rel in [('training_donor',donor),('unseen_test',held)]:
  path=R/folder/rel;inputs.append(path);a=read(path)
  assert all(b in master and master[b]['split']==split for b in a)
  sources.update(a)
  audit=path.parent/'coverage_audit.csv'
  if audit.exists():
   inputs.append(audit);coverage.update(read(audit))
 assert set(sources)|set(coverage)==set(master),(prefix,'missing basin coverage')
 loaded[prefix]=(sources,coverage)
header=['basin_id','catchment_split','country','basin_group','climate_class']
for prefix,*_ in models:header += [prefix+'_'+m for m in ['NSE','KGE','Bias_percent','N_scored_days','Status']]
rows=[]
for bid,m in sorted(master.items()):
 row=dict(basin_id=bid,catchment_split=m['split'],country=m['country'],basin_group=m['basin_group'],climate_class=m['climate_class'])
 for prefix,*_ in models:
  source,coverage=loaded[prefix];s=source.get(bid,{});c=coverage.get(bid,{})
  n=s.get('n',c.get('observed_test_days',''))
  if n!='':n=str(int(n))
  finite=lambda v: v not in ('',None) and math.isfinite(float(v))
  status=s.get('status') or c.get('status') or ('scored' if finite(s.get('NSE','')) else 'insufficient_observations' if n!='' and int(n)<2 else 'undefined_metrics')
  for metric,key in [('NSE','NSE'),('KGE','KGE'),('Bias_percent','bias_percent')]:
   value=s.get(key,'');row[prefix+'_'+metric]=value if finite(value) else ''
  row[prefix+'_N_scored_days']=n;row[prefix+'_Status']=status
 rows.append(row)
source_hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
out=B/'basin_lstm_performance_comparison.csv'
assert not out.exists(),'Output already exists; inspect before replacing.'
tmp=out.with_suffix('.tmp.csv')
with open(tmp,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=header);w.writeheader();w.writerows(rows)
tmp.replace(out)
with open(out,newline='') as f:check=list(csv.DictReader(f))
assert check==rows and len(check)==1426 and len(header)==25
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
for prefix,*_ in models:
 stats[prefix]={split:dict(rows=sum(r['catchment_split']==split for r in rows),scored=sum(r['catchment_split']==split and r[prefix+'_NSE']!='' for r in rows)) for split in ['training_donor','unseen_test']}
audit=dict(passed=True,rows=1426,columns=len(header),unique_basins=1426,training_donors=1326,unseen_basins=100,source_files_unchanged=True,source_sha256=source_hashes,coverage=stats,csv_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(B/'comparison_verification.json').write_text(json.dumps(audit,indent=2))
notes=['# Basin LSTM performance comparison','One basin per row. Catchment IDs are joined by exact text ID, never by row position. All 1,426 IDs are retained (1,326 training donors, 100 unseen catchments). Scores for donors are held-out temporal test scores, not training-fit scores.','## Models and periods']
for prefix,folder,donor,held,train,val,test in models:
 notes += [prefix+': '+folder+'. Training '+train+'; validation '+val+'; testing '+test+'. Seed 42.', 'Sources: '+folder+'/'+donor+' and '+folder+'/'+held+'.']
notes += ['The extended ERA5 model uses a different training and test period. Its score differences cannot be attributed solely to the meteorological source or attribute inputs. Its requested end_exclusive is 2023-01-04; forcing/observation availability determines the actual scored days for each basin.','## Columns','NSE and KGE: original saved metrics, ideal value 1. Bias_percent: 100*(sum(predicted)-sum(observed))/sum(observed), positive means overprediction, ideal 0. N_scored_days: number of paired valid observation/prediction days. Status preserves source explanations. Blank metrics are unavailable/undefined, never zero-filled. Counts of zero are retained when confirmed by a source metric/coverage audit.','catchment_split is training_donor or unseen_test. basin_group and climate_class come from the original split manifest. Climate labels are meteorological rain/snow proxies, not formal climate zones. The existing spatial-overlap limitation remains: 26 unseen IDs share polygon IDs with donors.','ERA5 meteorology-only has not yet been trained and is not represented by fabricated scores.','The CSV preserves source numeric precision. comparison_verification.json records coverage and source hashes.']
(B/'README.md').write_text((chr(10)+chr(10)).join(notes)+chr(10))
print('CREATED',out,'ROWS',len(rows),'COLUMNS',len(header));print(json.dumps(stats,indent=2))
