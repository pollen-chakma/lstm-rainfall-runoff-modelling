from pathlib import Path
import sys,json,time,subprocess,socket,traceback
BASE=Path(__file__).resolve().parent
PY=BASE.parent/'Model_A/.venv/bin/python'
def atomic(path,obj):
 t=path.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2)); t.replace(path)
def main():
 print('Waiting for verified full preparation and exact split',flush=True)
 while True:
  try: status=json.loads((BASE/'status.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError): time.sleep(5); continue
  if status.get('phase')=='failed': raise RuntimeError('Preparation failed: '+json.dumps(status))
  if status.get('phase')=='prepared': break
  time.sleep(5)
 config=json.loads((BASE/'experiment_config.json').read_text())
 assert len(config['training_basins'])==1326 and len(config['unseen_basins'])==100
 assert not set(config['training_basins'])&set(config['unseen_basins'])
 assert 'PASS:' in (BASE/'verification.log').read_text()
 print('Launching fresh H100 regional training: 1326 donors / 100 unseen',flush=True)
 with open(BASE/'training.log','w') as log:
  p=subprocess.Popen([str(PY),'-u',str(BASE/'train_unseen.py'),'--epochs','20','--batch','128','--per-basin','256','--validation-windows','128','--seed','42'],stdout=log,stderr=subprocess.STDOUT)
  atomic(BASE/'training_process.json',dict(pid=p.pid,host=socket.gethostname(),device='cuda:0',training_basins=1326,unseen_basins=100))
  code=p.wait()
  if code: raise RuntimeError('Training process exited '+str(code)+'; see training.log')
 print('REGIONAL EXPERIMENT COMPLETE',flush=True)
if __name__=='__main__':
 try: main()
 except Exception:
  atomic(BASE/'orchestrator_error.json',dict(error=traceback.format_exc())); raise
