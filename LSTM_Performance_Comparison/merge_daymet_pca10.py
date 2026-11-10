from pathlib import Path
import csv,json,collections,datetime,hashlib
R=Path('/home/pollen26/Desktop/LSTM_RR_Modelling'); D=R/'LSTM_Performance_Comparison'
P=D/'basin_lstm_performance_comparison.csv'; S=R/'Model A Lumped (Daymet + PCA10)'/'basin_model_performance.csv'
def read(p):
 with p.open(newline='') as f:
  r=csv.DictReader(f); return r.fieldnames,list(r)
heads,old=read(P); _,src=read(S)
assert len(old)==len(src)==1426
a={r['basin_id']:r for r in old}; b={r['basin_id']:r for r in src}
assert len(a)==len(b)==1426 and set(a)==set(b)
for k in a:
 assert all(a[k][c]==b[k][c] for c in ('seen_unseen','catchment_split'))
 assert b[k]['test_start']=='2005-01-01' and b[k]['test_end']=='2009-12-31'
fields={'NSE':'NSE','KGE':'KGE','RMSE':'RMSE','MAE':'MAE','Bias_percent':'bias_percent','N_scored_days':'n','Status':'status'}
prefix='Daymet_PCA10_Test2005_2009_'
added=[prefix+k for k in fields]; newheads=heads+[k for k in added if k not in heads]
rows=[dict(r) for r in old]
for r in rows:
 for suffix,key in fields.items(): r[prefix+suffix]=b[r['basin_id']][key]
tmp=P.with_suffix('.merge.tmp.csv')
with tmp.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=newheads); w.writeheader(); w.writerows(rows)
h,check=read(tmp); assert len(check)==1426 and h==newheads
for before,after in zip(old,check):
 assert all(before[k]==after[k] for k in heads if k not in added)
 assert all(after[prefix+s]==b[after['basin_id']][k] for s,k in fields.items())
backup=D/'backups'/('basin_lstm_performance_comparison_before_PCA10_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
backup.parent.mkdir(exist_ok=True); backup.write_bytes(P.read_bytes()); assert backup.read_bytes()==P.read_bytes()
tmp.replace(P)
info=dict(status='verified',rows=1426,columns_before=len(heads),columns_after=len(h),pca10_columns=added,seen_unseen=dict(collections.Counter(r['seen_unseen'] for r in check)),pca10_status=dict(collections.Counter(r[prefix+'Status'] for r in check)),backup=str(backup),source=str(S),source_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),original_columns_preserved=True)
(D/'pca10_merge_verification.json').write_text(json.dumps(info,indent=2)); print(json.dumps(info,indent=2))
