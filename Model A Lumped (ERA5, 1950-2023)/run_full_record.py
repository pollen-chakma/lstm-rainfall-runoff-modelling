from pathlib import Path
import json,time,subprocess,hashlib,shutil,traceback,os,fcntl
BASE=Path(__file__).resolve().parent; PY=BASE.parent/'Model_A/.venv/bin/python'
def phase(name,**kwargs):(BASE/'pipeline_status.json').write_text(json.dumps(dict(phase=name,**kwargs),indent=2))
def step(script,log,args=()):
 phase(script); print('START',script,flush=True)
 with open(BASE/log,'w') as f:subprocess.run([str(PY),str(BASE/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('FINISHED',script,flush=True)
def main():
 lock=open(BASE/'.pipeline.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('A pipeline is already running in this folder')
 if (BASE/'completion.json').exists():print('Already complete; original results preserved',flush=True); return
 phase('waiting_for_preparation')
 while True:
  s=json.loads((BASE/'status.json').read_text())
  if s.get('phase')=='prepared':break
  if s.get('phase')=='failed':raise RuntimeError('Preparation failed; inspect status.json and coverage_audit.csv')
  time.sleep(5)
 step('verify_protocol.py','verification.log')
 snap=BASE/'source_snapshot'; snap.mkdir(exist_ok=True)
 for p in BASE.iterdir():
  if p.suffix=='.py' or p.name in ['experiment_config.json','split_manifest.csv','training_basins.txt','unseen_test_basins.txt']:shutil.copy2(p,snap/p.name)
 for name in ['train_model_a.py','prepare_daymet.py']:shutil.copy2(BASE.parent/'Model_A'/name,snap/('pilot_'+name))
 (snap/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir() if p.is_file()},indent=2))
 step('train_unseen.py','training.log',('--resume',str(BASE/'run_seed42/last.pt')) if (BASE/'run_seed42/last.pt').exists() else ())
 step('evaluate_full_record.py','temporal_evaluation.log')
 hashes=json.loads((BASE/'prior_experiment_hashes.json').read_text()); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 run=BASE/'run_seed42'; unseen=json.loads((run/'unseen_evaluation_summary.json').read_text()); temporal=json.loads((run/'temporal_test_2015_latest/summary.json').read_text()); selected=json.loads((run/'summary.json').read_text())
 assert unseen['requested_basins']==100 and unseen['prediction_files']==100 and temporal['requested_basins']==1326 and temporal['prediction_files']==1326
 history=json.loads((run/'history.json').read_text()); assert len(history)==20
 final=dict(phase='complete',experiment='Model A Lumped (ERA5, 1950-2023)',training_period='1950-2009',validation_period='2010-2014',test_period='2015 through latest available early-2023 data',epochs=20,best_epoch=selected['best_epoch'],coverage=json.loads((BASE/'full_record_coverage_summary.json').read_text()),unseen=unseen,temporal=temporal,prior_experiments_preserved='SHA-256 PASS')
 (BASE/'completion.json').write_text(json.dumps(final,indent=2)); phase('complete'); print(json.dumps(final,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:phase('failed',error=traceback.format_exc()); (BASE/'pipeline_error.txt').write_text(traceback.format_exc()); raise
