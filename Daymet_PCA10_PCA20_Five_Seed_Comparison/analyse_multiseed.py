from pathlib import Path
import csv,json,hashlib,math,os,subprocess,shutil
import numpy as np
ROOT=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
B=Path(__file__).resolve().parent
O=B/'analysis_outputs';O.mkdir(exist_ok=True)
SEEDS=[42,43,44,45,46]
METRICS=['NSE','KGE','RMSE','MAE','bias_percent']
def read(p):return list(csv.DictReader(p.open()))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def number(s):
    try:return float(s)
    except (ValueError,TypeError):return float('nan')
def scores(y,q):
    good=np.isfinite(y)&np.isfinite(q);y=y[good];q=q[good];n=len(y)
    err=q-y
    out=dict(n=n,RMSE=float(np.sqrt(np.mean(err**2))) if n else np.nan,MAE=float(np.mean(np.abs(err))) if n else np.nan,NSE=np.nan,KGE=np.nan,bias_percent=np.nan)
    if n<2 or y.std()==0:return out
    out['NSE']=float(1-np.sum(err**2)/np.sum((y-y.mean())**2))
    r=np.corrcoef(y,q)[0,1] if q.std()>0 else np.nan
    out['KGE']=float(1-np.sqrt((r-1)**2+(q.std()/y.std()-1)**2+(q.mean()/y.mean()-1)**2)) if y.mean()!=0 else np.nan
    out['bias_percent']=float(100*(q.sum()-y.sum())/y.sum()) if y.sum()!=0 else np.nan
    return out
manifest=json.loads((B/'run_manifest.json').read_text())
runs={(r['pcs'],r['seed']):r for r in manifest}
baseline=read(Path(runs[10,42]['directory'])/'basin_model_performance.csv')
ids=[r['basin_id'] for r in baseline];assert len(ids)==len(set(ids))==1426
wide={r['basin_id']:dict(basin_id=r['basin_id'],seen_unseen=r['seen_unseen'],catchment_split=r['catchment_split'],test_start=r['test_start'],test_end=r['test_end'],n_scored_days=int(r['n'])) for r in baseline}
diagnostics=read(ROOT/'Poor_Basin_Performance_Analysis/analysis_outputs/all_basin_diagnostics.csv')
available=sorted({r['experiment'] for r in diagnostics})
preferred=[x for x in available if 'PCA10' in x];assert len(preferred)==1,(available,preferred)
diag={r['basin_id']:r for r in diagnostics if r['experiment']==preferred[0]};assert set(diag)==set(ids)
diagfields=['country','basin_group','climate_class','scored_coverage','test_q_mean_mm_day','test_q_max_mm_day','test_q_std_mm_day','test_q_cv','test_zero_flow_fraction','test_near_zero_fraction','test_flashiness','test_runoff_ratio','test_aridity_pet_over_prcp','test_snow_precip_fraction_proxy']
for bid in ids:
    assert diag[bid]['seen_unseen']==wide[bid]['seen_unseen']
    wide[bid].update({k:diag[bid][k] for k in diagfields})
verified=0;runinfo=[];observation_hashes={}
for pcs in [10,20]:
    perf={}
    reference_config=json.loads((Path(runs[pcs,42]['directory'])/'results_seed42/run_config.json').read_text())['config']
    for seed in SEEDS:
        run=runs[pcs,seed];d=Path(run['directory']);out=d/f'results_seed{seed}'
        if run['reused']:assert hashlib.sha256((d/'basin_model_performance.csv').read_bytes()).hexdigest()==run['performance_sha256'],'Reused seed-42 metrics changed after preparation'
        history=json.loads((out/'history.json').read_text());assert [h['epoch'] for h in history]==list(range(1,21))
        cfg=json.loads((out/'run_config.json').read_text())['config']
        assert cfg['seed']==seed and cfg['epochs']==20 and cfg['hidden']==64 and cfg['windows_per_basin']==256
        normalized=dict(cfg,seed=42);assert normalized==reference_config
        if seed!=42:
            nb=json.loads(Path(run['notebook']).read_text());code=[c for c in nb['cells'] if c['cell_type']=='code']
            assert len(code)==7 and all(c['execution_count'] is not None for c in code)
            assert not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))
        fit=json.loads((out/'fit_metadata.json').read_text());assert len(fit['fitted_ids'])==1326 and len(fit['excluded_ids'])==100
        assert set(fit['fitted_ids'])=={i for i in ids if wide[i]['seen_unseen']=='Seen'}
        assert set(fit['excluded_ids'])=={i for i in ids if wide[i]['seen_unseen']=='Unseen'}
        perf[seed]={r['basin_id']:r for r in read(d/'basin_model_performance.csv')};assert set(perf[seed])==set(ids)
        runinfo.append(dict(model=f'PCA{pcs}',seed=seed,reused_seed42=run['reused'],best_epoch=json.loads((out/'summary.json').read_text())['best_epoch'],epochs=20,csv_sha256=json.loads((out/'run_config.json').read_text())['csv_sha256']))
    # All seeds of a setup must read exactly the original CSV.
    assert len({x['csv_sha256'] for x in runinfo if x['model']==f'PCA{pcs}'})==1
    ensemble_file=O/f'PCA{pcs}_ensemble_daily_predictions.csv'
    with ensemble_file.open('w',newline='') as ef:
        ew=csv.writer(ef);ew.writerow(['basin_id','seen_unseen','date','observed_mm_day','predicted_mm_day'])
        for j,bid in enumerate(ids):
            dates=None;observed=None;predictions=[]
            tag='donor_temporal' if wide[bid]['seen_unseen']=='Seen' else 'unseen100'
            for seed in SEEDS:
                d=Path(runs[pcs,seed]['directory']);rows=read(d/f'results_seed{seed}'/tag/(bid+'_predictions.csv'))
                ds=np.array([r['date'] for r in rows]);y=np.array([r['observed_mm_day'] for r in rows],dtype='float32');q=np.array([r['predicted_mm_day'] for r in rows],dtype='float64')
                assert len(ds)==len(set(ds)) and np.isfinite(q).all()
                assert np.all(ds>='2005-01-01') and np.all(ds<='2009-12-31')
                if dates is None:
                    dates=ds;observed=y
                    observation_digest=hashlib.sha256(ds.tobytes()+y.tobytes()).hexdigest()
                    if pcs==10:observation_hashes[bid]=observation_digest
                    else:assert observation_digest==observation_hashes[bid],('PCA10/PCA20 test observations differ',bid)
                else:assert np.array_equal(ds,dates) and np.array_equal(y,observed,equal_nan=True)
                saved=perf[seed][bid];recomputed=scores(y,q)
                assert recomputed['n']==int(saved['n'])==wide[bid]['n_scored_days']
                for m in METRICS:
                    assert np.allclose(recomputed[m],number(saved[m]),rtol=2e-5,atol=2e-5,equal_nan=True),(pcs,seed,bid,m,recomputed[m],saved[m])
                    wide[bid][f'PCA{pcs}_seed{seed}_{m}']=number(saved[m])
                predictions.append(q);verified+=1
            ensemble=np.stack(predictions).mean(axis=0)
            es=scores(observed,ensemble)
            for m in METRICS:wide[bid][f'PCA{pcs}_ensemble_{m}']=es[m]
            ew.writerows((bid,wide[bid]['seen_unseen'],date,float(y),float(q)) for date,y,q in zip(dates,observed,ensemble))
            if (j+1)%200==0:print('VERIFIED AND ENSEMBLED',pcs,j+1,flush=True)
write(O/'run_provenance.csv',runinfo)
write(B/'multiseed_basin_analysis_dataset.csv',[wide[i] for i in ids])
(O/'verification.json').write_text(json.dumps(dict(status='passed',basin_seed_predictions_verified=verified,expected=1426*10,all_seeds_same_ids_years_scored_days=True,all_daily_metrics_recomputed=True,ensemble_method='arithmetic mean of five daily discharge predictions, then metrics recalculated',data_sha256=hashlib.sha256((B/'multiseed_basin_analysis_dataset.csv').read_bytes()).hexdigest()),indent=2))
# The analysis notebook and the single basin CSV reproduce all comparison tables and figures.
analysis_source=(B/'report_analysis.py').read_text()
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Basin Performance Diagnostics',language='python',name='basin_diagnostics')),cells=[dict(cell_type='markdown',metadata={},source=['# Five-seed Daymet PCA10 versus PCA20 comparison\n','\n','This notebook reads only multiseed_basin_analysis_dataset.csv beside it. The CSV contains per-seed scores, metrics recalculated from mean daily discharge predictions, and observed hydrological diagnostics. It reproduces comparison tables and figures without training or accessing the original experiment folders. Use each seed notebook to rerun model training.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=analysis_source.splitlines(keepends=True))])
(B/'Five_Seed_Comparison_Analysis.ipynb').write_text(json.dumps(notebook))
exec(compile(analysis_source,str(B/'report_analysis.py'),'exec'),{'__name__':'__main__','__file__':str(B/'report_analysis.py')})
runner_python=ROOT/'Model_A/.venv/bin/python'
notebook_path=B/'Five_Seed_Comparison_Analysis.ipynb'
with (B/'analysis_notebook_execution.log').open('w') as log:
    subprocess.run([str(runner_python),str(B/'run_all.py'),str(notebook_path)],stdout=log,stderr=subprocess.STDOUT,check=True)
repro=B/'two_file_reproduction';repro.mkdir(exist_ok=True)
shutil.copy2(B/'multiseed_basin_analysis_dataset.csv',repro/'multiseed_basin_analysis_dataset.csv')
shutil.copy2(notebook_path,repro/notebook_path.name)
with (B/'two_file_reproduction.log').open('w') as log:
    subprocess.run([str(runner_python),str(B/'run_all.py'),str(repro/notebook_path.name)],stdout=log,stderr=subprocess.STDOUT,check=True)
tables=['per_seed_group_performance.csv','repeatability_summary.csv','paired_seed_comparison.csv','basin_improvement_consistency.csv','ensemble_performance_summary.csv','hydrological_subgroup_comparison.csv','consistency_distribution.csv']
for name in tables:assert (O/name).read_bytes()==(repro/'analysis_outputs'/name).read_bytes(),name
from PIL import Image
figures=[f'{metric}_{kind}.png' for metric in ['NSE','KGE'] for kind in ['five_seed_repeatability','basin_consistency']]
for name in figures:
    with Image.open(O/name) as a,Image.open(repro/'analysis_outputs'/name) as b:
        assert np.array_equal(np.asarray(a),np.asarray(b)),name
(O/'two_file_reproduction_verification.json').write_text(json.dumps(dict(status='passed',input_files=['Five_Seed_Comparison_Analysis.ipynb','multiseed_basin_analysis_dataset.csv'],tables_identical=tables,figures_pixel_identical=figures),indent=2))
print('ANALYSIS COMPLETE',flush=True)
