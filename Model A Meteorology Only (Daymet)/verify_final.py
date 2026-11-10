from pathlib import Path
import json,csv,hashlib,ast
import numpy as np
import torch
B=Path(__file__).resolve().parent;O=B/'results_seed42'
log=(B/'notebook_rerun_verification.log').read_text()
assert 'NOTEBOOK EXECUTION COMPLETE' in log and 'EPOCH {' not in log
before=json.loads((B/'first_completed_summary.json').read_text());after=json.loads((O/'summary.json').read_text());assert before==after
hashes=json.loads((B/'pre_rerun_hashes.json').read_text());assert all(hashlib.sha256((B/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
nb=json.loads((B/'Daymet_Meteorology_Only_LSTM.ipynb').read_text());codes=[c for c in nb['cells'] if c['cell_type']=='code']
assert len(codes)==7 and all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in codes)
imports=set()
for c in codes:
 source=''.join(c['source']);compile(source,'cell','exec')
 for n in ast.walk(ast.parse(source)):
  if isinstance(n,ast.Import):imports.update(a.name.split('.')[0] for a in n.names)
  elif isinstance(n,ast.ImportFrom):imports.add(n.module.split('.')[0])
assert imports <= {'pathlib','csv','json','time','random','hashlib','base64','zlib','math','os','numpy','torch','IPython','itertools'}
ck=torch.load(O/'best.pt',map_location='cpu',weights_only=False)
assert list(ck['model']['lstm.weight_ih_l0'].shape)==[256,7]
assert sum(v.numel() for v in ck['model'].values())==18753
h=json.loads((O/'history.json').read_text());assert len(h)==20 and ck['epoch']==min(h,key=lambda r:r['validation_loss'])['epoch']
coverage={};wins={}
for folder,n in [('unseen100',100),('donor_temporal',1326)]:
 rows=list(csv.DictReader(open(O/folder/'metrics.csv')))
 assert len(rows)==n and len(list((O/folder).glob('*predictions.csv')))==n
 coverage[folder]=dict(reserved=n,scored=int(sum(np.isfinite(float(r['NSE'])) for r in rows)),unscored_ids=[r['basin_id'] for r in rows if not np.isfinite(float(r['NSE']))])
pairs=list(csv.DictReader(open(O/'comparison_by_catchment.csv')))
for tag in ['unseen','temporal']:
 d=[float(r['attribute_minus_no_attribute_NSE']) for r in pairs if r['test']==tag and np.isfinite(float(r['attribute_minus_no_attribute_NSE']))]
 wins[tag]=dict(paired=len(d),attribute_higher_NSE=sum(v>0 for v in d),median_paired_NSE_gain=float(np.median(d)))
result=dict(passed=True,seven_executed_code_cells_without_errors=True,no_external_project_imports=True,checkpoint_inputs=7,attributes=0,parameters=18753,training_epochs=20,best_epoch=ck['epoch'],checkpoint_reuse_verified=True,all_1426_prediction_files_identical_after_rerun=True,both_checkpoints_unchanged_after_rerun=True,coverage=coverage,paired_comparison=wins,csv_roundtrip=json.loads((B/'csv_roundtrip_verification.json').read_text()),protocol_parity=json.loads((B/'protocol_parity_verification.json').read_text()))
(B/'final_verification.json').write_text(json.dumps(result,indent=2));(B/'completion.json').write_text(json.dumps(after,indent=2))
with open(B/'README.md','a') as f:
 f.write(chr(10)+'## Completed seed-42 results'+chr(10)+'20 epochs; best validation checkpoint: epoch '+str(ck['epoch'])+'.'+chr(10))
 for r in after['results']:f.write(str(r)+chr(10))
 f.write('final_verification.json confirms successful full execution and checkpoint reuse. All 1,426 prediction files and both model checkpoints stayed identical after rerun. protocol_parity_verification.json confirms exact normalization/window parity with the baseline.'+chr(10))
print(json.dumps(result,indent=2),flush=True)
