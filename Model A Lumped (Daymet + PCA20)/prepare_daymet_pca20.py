"""Prepare a distinct, protocol-matched Daymet + PCA20 experiment on Nibi."""
from pathlib import Path
import ast, collections, csv, hashlib, itertools, json, math, re, time

R=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
OLD=R/'Model A Lumped (Daymet + PCA10)'
B=R/'Model A Lumped (Daymet + PCA20)'
B.mkdir(exist_ok=True)
NB=B/'Daymet_PCA20_LSTM.ipynb'
CSV=B/'daymet_pca20_discharge_1980_2009.csv'
BASE=R/'Model A Meteorology Only (Daymet)'/'daymet_forcing_discharge_1980_2009.csv'
PCA=R/'Subbasin_Heterogeneity_PCA'/'basin_pca20_mean_std_features.csv'
assert not CSV.exists() and not NB.exists(), 'Existing PCA20 files: inspect saved state before retrying.'
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
pc=[f'PC{i:02d}_{s}' for i in range(1,21) for s in ('mean','std')]
rows=read(PCA); pr={r['basin_id']:r for r in rows}
oldpc={r['basin_id']:r for r in read(R/'Subbasin_Heterogeneity_PCA/basin_pca10_mean_std_features.csv')}
assert len(pr)==len(rows)==len(oldpc)==1426 and set(pr)==set(oldpc)
for bid,r in pr.items():
    assert all(math.isfinite(float(r[k])) for k in pc)
    for k,v in oldpc[bid].items():
        if k.startswith('PC'):assert math.isclose(float(v),float(r[k]),rel_tol=1e-10,abs_tol=1e-10),(bid,k)
        else:assert v==r[k],(bid,k)
assert collections.Counter(r['catchment_split'] for r in rows)=={'training_donor':1326,'unseen_test':100}
n=json.loads((OLD/'Daymet_PCA10_LSTM.ipynb').read_text())
baseline=json.loads(json.dumps(n))
def source(c):return ''.join(c['source'])
for i,c in enumerate(n['cells']):
    s=source(c).replace('PCA10','PCA20').replace('pca10','pca20')
    if c['cell_type']=='code':
        c['outputs']=[];c['execution_count']=None
        if i==2:
            s,count=re.subn(r'^STATIC_FEATURES=.*$', 'STATIC_FEATURES='+repr(pc),s,flags=re.M);assert count==1
        if i==4:
            for a,b in [('[11:31]','[11:51]'),('[11:34]','[11:54]')]:assert a in s;s=s.replace(a,b)
            s=re.sub(r'\[(31|32|33|34)\]',lambda m:'['+str(int(m[1])+20)+']',s)
        if i in (6,8):s=re.sub(r'\b27\b','47',s)
        if i==8:assert '23873' in s;s=s.replace('23873','28993')
        compile(s,f'PCA20 cell {i}','exec')
    c['source']=s.splitlines(keepends=True)
intro='''# Model A Lumped — Daymet + PCA20 subbasin summaries

This controlled experiment changes the static representation from PCA10 to PCA20.
Inputs: the same seven Daymet meteorological variables plus 40 static features:
area-weighted means and standard deviations of PC01–PC20. Target: discharge in mm/day.
The shared PCA20 features are already fitted using donor subbasins; PCA is not refitted here.
The first ten components and basin metadata match the PCA10 export.

## Preserved experiment settings
Exactly 1,326 training-donor catchments and 100 unseen catchments, with the original IDs.
Train 1980–1999; validate 2000–2004; test 2005–2009 in both Seen and Unseen groups.
One LSTM layer, hidden size 64, linear discharge head, 365-day input window, Adam 0.001,
batch 128, 256 sampled training windows/basin/epoch, 20 epochs, seed 42, gradient clipping 1.
Validation uses at most 128 fixed windows per donor basin. Loss, sampling, preprocessing,
missing-data policy and window endpoint convention are copied from PCA10.
There are 47 input channels and 28,993 trainable parameters; the additional inputs necessarily
increase the parameter count while the hidden-layer setup stays unchanged.

## Read → prepare → train → validate → test → check performance
Run all cells in order using the Model A PyTorch (GPU) kernel. This notebook reads only
`daymet_pca20_discharge_1980_2009.csv`, placed beside it. NumPy, PyTorch, IPython and
Jupyter kernel support must be installed; no pandas dependency is needed.
The static scaler is fitted once per donor basin; meteorological and flow scalers use only
donor training years. Unseen targets do not fit model/scalers. Validation alone selects
the checkpoint. Earlier forcing can provide validation/test context; discharge is never input.
Geographic overlap between some donor and held-out catchments remains as documented previously.

Existing `results_seed42/last.pt` resumes only if configuration and CSV hashes match.
Use a different RUN_NAME for a new independent training run. GPU arithmetic can vary slightly
across hardware/software despite a fixed seed. The single-subbasin flag is metadata, not an input.
'''
n['cells'][0]['source']=intro.splitlines(keepends=True)
def cfg(s):
    tree=ast.parse(s)
    node=next(z.value for z in tree.body if isinstance(z,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CFG' for t in z.targets))
    return {k.arg:ast.literal_eval(k.value) for k in node.keywords if k.arg not in ('features','static_features')}
assert cfg(source(n['cells'][2]))==cfg(source(baseline['cells'][2]))
assert cfg(source(n['cells'][2]))['epochs']==20
NB.write_text(json.dumps(n,indent=1))
executor=(OLD/'execute_notebook.py').read_text().replace('Daymet_PCA10_LSTM.ipynb','Daymet_PCA20_LSTM.ipynb')
(B/'execute_notebook.py').write_text(executor)
plan=dict(status='notebook_ready',settings=cfg(source(n['cells'][2])),
          dynamic_inputs=7,static_inputs=40,input_channels=47,parameters=28993,
          donor_basins=1326,unseen_basins=100,first_10_PCs_match_PCA10=True,
          baseline_notebook_sha256=hashlib.sha256((OLD/'Daymet_PCA10_LSTM.ipynb').read_bytes()).hexdigest(),
          pca20_sha256=hashlib.sha256(PCA.read_bytes()).hexdigest(),pca_source=str(PCA),
          meteorology_discharge_source=str(BASE),only_notebook_input=CSV.name)
(B/'experiment_provenance.json').write_text(json.dumps(plan,indent=2))
(B/'README.md').write_text(intro+'\nOutputs: basin_model_performance.csv (all 1,426 basins with Seen/Unseen and missing-score status), model_performance_summary.csv, results_seed42/ (scalers, checkpoints, history and daily predictions), completion.json. Earlier experiments remain unchanged. Preparation and verification scripts are audit helpers; rerunning the model requires only the notebook and combined CSV, plus the Python environment.\n')
print('NOTEBOOK AND PROTOCOL CHECKS PASSED',flush=True)
tmp=CSV.with_suffix('.tmp.csv');assert not tmp.exists(),'Inspect partial CSV before retrying.'
counts=collections.Counter();start=time.time()
with BASE.open(newline='') as f,tmp.open('w',newline='') as g:
    reader=csv.reader(f);h=next(reader)
    assert h==['basin_id','split','date','prcp','tmax','tmin','srad','vp','dayl','pet','q_obs_mm_day']
    writer=csv.writer(g);writer.writerow(h+pc+['seen_unseen','n_subbasins','single_subbasin_flag','time_split'])
    previous=None;extra=None;total=0
    for line in reader:
        bid,split,date=line[:3]
        if bid!=previous:
            r=pr[bid];assert r['catchment_split']==split
            extra=[r[k] for k in pc]+[r['seen_unseen'],r['n_subbasins'],r['single_subbasin_flag']];previous=bid
        period='train' if date<'2000-01-01' else ('validation' if date<'2005-01-01' else 'test')
        writer.writerow(line+extra+[period]);counts[bid]+=1;total+=1
        if total%1000000==0:print('EXPORTED',total,flush=True)
assert set(counts)==set(pr) and set(counts.values())=={10958} and sum(counts.values())==15626108
print('EXPORT COMPLETE; VERIFYING EVERY ROW',flush=True)
verified=0
with BASE.open(newline='') as f,tmp.open(newline='') as g:
    a=csv.reader(f);b=csv.reader(g);h=next(a);nh=next(b)
    assert len(nh)==55
    for x,y in itertools.zip_longest(a,b):
        assert x is not None and y is not None and y[:11]==x
        r=pr[x[0]]
        assert y[11:54]==[r[k] for k in pc]+[r['seen_unseen'],r['n_subbasins'],r['single_subbasin_flag']]
        assert y[54]==('train' if x[2]<'2000-01-01' else 'validation' if x[2]<'2005-01-01' else 'test')
        verified+=1
assert verified==15626108 and not CSV.exists();tmp.replace(CSV)
plan.update(status='dataset_verified',rows=verified,columns=55,days_per_basin=10958,preparation_seconds=time.time()-start,
            original_meteorology_discharge_columns_unchanged=True,all_static_values_verified=True)
(B/'dataset_verification.json').write_text(json.dumps(plan,indent=2))
print('PREPARATION COMPLETE',json.dumps(plan),flush=True)
