from pathlib import Path
import json,time,subprocess,csv,hashlib,shutil,traceback
BASE=Path(__file__).resolve().parent; DAY=BASE.parent/'Model A Lumped (Daymet)'; PY=BASE.parent/'Model_A/.venv/bin/python'
def writecsv(p,rs):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
def step(script,log):
 print('START',script,flush=True)
 with open(BASE/log,'w') as f: subprocess.run([str(PY),str(BASE/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('FINISHED',script,flush=True)
def main():
 while True:
  state=json.loads((BASE/'status.json').read_text()) if (BASE/'status.json').exists() else {}
  if state.get('phase')=='prepared': break
  if state.get('phase')=='failed': raise RuntimeError('Preparation failed; inspect status.json')
  time.sleep(5)
 manifest=list(csv.DictReader(open(BASE/'split_manifest.csv'))); md={r['basin_id']:r for r in manifest}; coverage=list(csv.DictReader(open(BASE/'coverage_audit.csv')))
 for r in coverage:r['basin_group']=md[r['basin_id']]['basin_group']
 writecsv(BASE/'coverage_audit.csv',coverage)
 step('verify_protocol.py','verification.log')
 snap=BASE/'source_snapshot'; snap.mkdir(exist_ok=True)
 for name in ['prepare_era5.py','train_unseen.py','verify_protocol.py','evaluate_temporal.py','fast_dates.py','experiment_config.json','split_manifest.csv','training_basins.txt','unseen_test_basins.txt','run_all_era5.py']: shutil.copy2(BASE/name,snap/name)
 for name in ['train_model_a.py','prepare_daymet.py']: shutil.copy2(BASE.parent/'Model_A'/name,snap/('pilot_'+name))
 (snap/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir() if p.is_file()},indent=2))
 step('train_unseen.py','training.log')
 step('summarize_unseen.py','final_verification.log')
 step('evaluate_temporal.py','temporal_evaluation.log')
 hashes=json.loads((BASE/'daymet_preservation_hashes.json').read_text()); assert all(hashlib.sha256((DAY/k).read_bytes()).hexdigest()==v for k,v in hashes.items()),'Daymet preservation check failed'
 u=json.loads((BASE/'run_seed42/summary.json').read_text()); t=json.loads((BASE/'run_seed42/temporal_test_2005_2009/summary.json').read_text()); assert u['unseen_basins']==100 and t['requested_basins']==1326 and t['prediction_files']==1326
 comparisons=[]
 for source,folder in [('Daymet',DAY),('ERA5',BASE)]:
  for name,prefix in [('unseen100','run_seed42/summary.json'),('trained_catchments_temporal','run_seed42/temporal_test_2005_2009/summary.json')]:
   s=json.loads((folder/prefix).read_text()); comparisons.append(dict(forcing=source,test=name,scored_basins=s.get('unseen_basins',s.get('scored_basins')),median_NSE=s['median_NSE'],median_KGE=s['median_KGE']))
 writecsv(BASE/'daymet_era5_comparison.csv',comparisons)
 (BASE/'completion.json').write_text(json.dumps(dict(phase='complete',forcing='ERA5',unseen=u,temporal=t,daymet_preservation='PASS: original split, checkpoints, preprocessing and summary files unchanged',comparison=comparisons),indent=2))
 print('ALL ERA5 STAGES COMPLETE',json.dumps(comparisons),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:(BASE/'pipeline_error.txt').write_text(traceback.format_exc()); raise
