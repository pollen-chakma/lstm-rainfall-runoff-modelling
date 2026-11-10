from pathlib import Path
import sys,json,csv,html
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'Model_A'))
from inspect_model_a import table,chart

def rows(name):
 p=BASE/name
 return list(csv.DictReader(open(p))) if p.exists() else []
def report(name):
 status=json.loads((BASE/'status.json').read_text()) if (BASE/'status.json').exists() else {'phase':'not_started'}
 if name=='status':
  out='<h3>Live experiment status</h3>'+table(['Field','Value'],[[k,v] for k,v in status.items()])
  if (BASE/'experiment_config.json').exists():
   c=json.loads((BASE/'experiment_config.json').read_text())
   out+=table(['Protocol','Value'],[['Training donors',len(c['training_basins'])],['Unseen catchments',len(c['unseen_basins'])],['Forcing','7 daily Daymet variables'],['Attributes',', '.join(c['static'])],['Train / donor validation / unseen test','1980–1999 / 2000–2004 / 2005–2009'],['Task','Same-day rainfall–runoff simulation'],['Overlap screen',str(c['held_basins_with_shared_ids'])+' held-out basins share polygon IDs with donors; exact split retained']])
  else: out+='<p>Split will be frozen after all 1,426 source files pass preparation. Training launches automatically only after exact-count and leakage checks pass.</p>'
  return out
 if name=='split':
  s=rows('stratum_counts.csv'); m=rows('split_manifest.csv')
  if not m:return '<p>Full coverage audit and stratified selection are still running.</p>'
  held=[r for r in m if r['split']=='unseen_test']
  out='<p>Fixed split seed: 20261031. Climate types are cold-weather precipitation proxies derived from pre-2000 Daymet data: rain &lt;0.1; mixed 0.1–&lt;0.4; snow ≥0.4. Every nonempty eligible group/climate/country stratum is represented, then quotas are proportional. These are not formal climate zones.</p>'
  out+=table(['Split','Basin group','Climate proxy','Country','Count'],[[r['split'],r['basin_group'],r['climate_class'],r['country'],r['count']] for r in s])
  out+='<h3>The 100 unseen catchments</h3>'+table(['Basin','Group','Climate proxy','Country','Cold-precipitation fraction','Valid test windows'],[[r['basin_id'],r['basin_group'],r['climate_class'],r['country'],float(r['cold_precip_fraction']),r['test_windows']] for r in held])
  out+='<p>The full 1,326 donor IDs are in training_basins.txt. All assignments are in split_manifest.csv. Shared polygon IDs are a screening proxy, not an exhaustive geometric overlap test. Overlapping donor basins were retained to preserve the requested 1,326/100 split.</p>'
  return out
 if name=='coverage':
  m=rows('coverage_audit.csv')
  if not m:return '<p>Audit in progress; completed '+str(status.get('completed',0))+' of 1,426.</p>'
  out=[]
  for g in sorted({r['basin_group'] for r in m}):
   subset=[r for r in m if r['basin_group']==g]
   out.append([g,len(subset),sum(r['status']=='ok' for r in subset),min(int(r['train_windows']) for r in subset),int(np.median([int(r['train_windows']) for r in subset])),min(int(r['validation_windows']) for r in subset),min(int(r['test_windows']) for r in subset)])
  return '<p>Daily intervals and units are checked. Missing-weather sequences, negative flows and partial-day observations are excluded. Estimated and ice-affected observations are retained, consistent with the pilot. Missing static inputs are imputed with donor-only medians during training.</p>'+table(['Group','Catchments','Source audit OK','Minimum train windows','Median train windows','Minimum validation windows','Minimum test windows'],out)
 if name=='training':
  p=BASE/'run_seed42/history.json'
  if not p.exists():return '<p>Training has not completed its first epoch. See status.json and training.log.</p>'
  h=json.loads(p.read_text())
  return '<p>Each epoch samples 256 valid windows from each of the 1,326 donors: 339,456 samples per epoch, batch size 128. Checkpoint validation uses a fixed, evenly spaced set of up to 128 valid days per donor. No unseen catchments are used. Complete optimizer/RNG checkpoints: run_seed42/last.pt and best.pt.</p>'+chart('Regional Model A loss',[('Training',np.array([x['training_loss'] for x in h]),'#2563eb'),('Validation',np.array([x['validation_loss'] for x in h]),'#ea580c')],'Normalized loss')+table(['Epoch','Training loss','Validation loss','Validation donors','Seconds'],[[r['epoch'],r['training_loss'],r['validation_loss'],r['validation_donors'],r['elapsed_seconds']] for r in h])
 if name=='results':
  m=rows('run_seed42/unseen_metrics.csv')
  if not m:return '<p>Unseen performance is not yet available. These catchments are evaluated only after training and validation-based checkpoint selection finish.</p>'
  summary=json.loads((BASE/'run_seed42/summary.json').read_text())
  out=table(['Summary','Value'],[[k,v] for k,v in summary.items()])
  groups=rows('run_seed42/unseen_group_metrics.csv')
  out+='<h3>Performance by basin group, climate proxy and country</h3>'+table(['Grouping','Category','Catchments','Median NSE','Median KGE'],[[r['grouping'],r['category'],r['basins'],float(r['median_NSE']),float(r['median_KGE'])] for r in groups])
  screen=rows('run_seed42/overlap_screen_metrics.csv')
  if screen:
   out+='<h3>Shared-polygon screen sensitivity</h3><p>No shared IDs does not prove geometric independence; this is a non-exhaustive screen. The requested exact 1,326/100 split retains overlapping drainage areas.</p>'+table(['Screen group','Catchments','Median NSE','Median KGE'],[[r['screen_group'],r['catchments'],float(r['median_NSE']),float(r['median_KGE'])] for r in screen])
  out+='<h3>All 100 unseen catchments</h3>'+table(['Unseen basin','Group','Climate proxy','Country','Scored days','NSE','KGE','Volume bias [%]'],[[r['basin_id'],r['basin_group'],r['climate_class'],r['country'],r['n'],float(r['NSE']),float(r['KGE']),float(r['bias_percent'])] for r in m])
  return out
 raise ValueError(name)
if __name__=='__main__':print(json.dumps({'html':report(sys.argv[1])}))
