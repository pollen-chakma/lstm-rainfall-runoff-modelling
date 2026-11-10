from pathlib import Path
import csv, json, math, statistics, hashlib, shutil, os
from datetime import datetime, timezone

ROOT=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
BASE=ROOT/'LSTM_Performance_Comparison'
SOURCE=ROOT/'Model A Lumped (Daymet + PCA10 EA-LSTM H256 Dropout02 Adaptive50)'/'outputs/basin_model_performance.csv'
TARGET=BASE/'basin_lstm_performance_comparison.csv'
SUMMARY=BASE/'PCA20_updated_boxplots/model_summary_statistics.csv'
PREFIX='Daymet_PCA10_EALSTM_H256_Dropout02_Adaptive50_Test2005_2009_'
LABEL='Daymet + PCA10 mean/std (EA-LSTM H256, dropout 0.2, adaptive50)'
MAPPING={'NSE':'NSE','KGE':'KGE','RMSE':'RMSE','MAE':'MAE','Bias_percent':'bias_percent','N_scored_days':'n','Status':'status'}
def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f);return r.fieldnames,list(r)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,cols,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def quantile(a,q):
    pos=(len(a)-1)*q;i=math.floor(pos);j=math.ceil(pos)
    return a[i]+(a[j]-a[i])*(pos-i)
def finite(s):
    try:return math.isfinite(float(s))
    except (ValueError,TypeError):return False

oldcols,old=read(TARGET);scols,summary=read(SUMMARY);_,source=read(SOURCE)
original_hash={str(p):digest(p) for p in [TARGET,SUMMARY,SOURCE]}
assert len(old)==len(source)==1426
assert len({r['basin_id'] for r in old})==len({r['basin_id'] for r in source})==1426
lookup={r['basin_id']:r for r in source}
assert set(lookup)=={r['basin_id'] for r in old}
newcols=[PREFIX+k for k in MAPPING]
assert not any(c.startswith(PREFIX) for c in oldcols),'EA columns already present; inspect before replacing'
assert not any(r['model']==LABEL for r in summary),'EA summary already present'
assert sum(r['seen_unseen']=='Seen' for r in old)==1326
assert sum(r['seen_unseen']=='Unseen' for r in old)==100
merged=[]
for r in old:
    s=lookup[r['basin_id']]
    assert s['seen_unseen']==r['seen_unseen']
    assert s['catchment_split']==r['catchment_split']
    assert s['test_start']=='2005-01-01' and s['test_end']=='2009-12-31'
    x=dict(r)
    for target,src in MAPPING.items():x[PREFIX+target]=s[src]
    merged.append(x)
add=[]
for group in ['Seen','Unseen']:
    for metric in ['NSE','KGE']:
        a=sorted(float(r[metric]) for r in source if r['seen_unseen']==group and finite(r[metric]))
        row=dict(model=LABEL,test_period='2005–2009',group=group,metric=metric,n=len(a),mean=statistics.mean(a),median=statistics.median(a),q1=quantile(a,.25),q3=quantile(a,.75),std=statistics.stdev(a),minimum=min(a),maximum=max(a),negative=sum(v<0 for v in a),below_minus_one=sum(v< -1 for v in a))
        assert set(row)==set(scols)
        add.append(row)
assert [(r['n'],r['negative']) for r in add]==[(1317,56),(1317,76),(100,10),(100,17)]
updated=summary+add
updated.sort(key=lambda r:float(r['median']),reverse=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=BASE/'backups'/('before_EALSTM_H256_'+stamp);backup.mkdir(parents=True)
shutil.copy2(TARGET,backup/TARGET.name);shutil.copy2(SUMMARY,backup/SUMMARY.name)
temp1=TARGET.with_suffix('.ea_pending.csv');temp2=SUMMARY.with_suffix('.ea_pending.csv')
write(temp1,oldcols+newcols,merged);write(temp2,scols,updated)
c1,r1=read(temp1);c2,r2=read(temp2)
assert c1==oldcols+newcols and c2==scols
assert len(r1)==1426 and len(r2)==len(summary)+4
assert [{k:r[k] for k in oldcols} for r in r1]==old
old_summary={tuple(r.items()) for r in summary}
assert old_summary.issubset({tuple(r.items()) for r in r2})
for r in r1:
    assert all(r[PREFIX+k]==lookup[r['basin_id']][v] for k,v in MAPPING.items())
assert all(digest(Path(p))==h for p,h in original_hash.items()),'Files changed during preparation'
os.replace(temp1,TARGET);os.replace(temp2,SUMMARY)
assert read(TARGET)==(c1,r1) and read(SUMMARY)==(c2,r2)
verification=dict(status='passed',source=str(SOURCE),source_sha256=digest(SOURCE),basin_rows=len(r1),basin_columns=len(c1),summary_rows=len(r2),added_columns=newcols,added_summary_rows=add,existing_values_and_order_preserved=True,seen_unseen_preserved=True,backup=str(backup),target_sha256=digest(TARGET),summary_sha256=digest(SUMMARY))
(BASE/'EALSTM_H256_merge_verification.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(dict(status='passed',basin_rows=len(r1),basin_columns=len(c1),summary_rows=len(r2),backup=str(backup),added_summary_rows=add)),flush=True)
