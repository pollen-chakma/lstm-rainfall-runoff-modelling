from pathlib import Path
import json,csv,sys,html,statistics,math
BASE=Path(__file__).resolve().parent

def read(name,default=None):
 p=BASE/name; return json.loads(p.read_text()) if p.exists() else default

def rows(name):
 p=BASE/name; return list(csv.DictReader(open(p))) if p.exists() else []

def table(headers,data):
 def cell(v):
  if isinstance(v,float):return f'{v:.4f}' if math.isfinite(v) else 'not available'
  return str(v)
 return '<table style="border-collapse:collapse;font-size:13px" border="1" cellpadding="6"><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(cell(x))+'</td>' for x in row)+'</tr>' for row in data)+'</table>'

def report(section):
 c=read('experiment_config.json',{})
 if section=='status':
  out='<h2>ERA5 full-record experiment</h2>'+table(['Setting','Value'],[['Training period','1950–2009'],['Validation period','2010–2014'],['Test period','2015 through latest available early-2023 data'],['Training catchments',len(c.get('training_basins',[]))],['Unseen catchments',len(c.get('unseen_basins',[]))],['Inputs',', '.join(c.get('features',[]))],['Attributes',', '.join(c.get('static',[]))],['Model','LSTM 64 hidden units, 365-day lookback'],['Training','20 epochs; seed 42; batch 128; 256 windows per donor per epoch'],['Validation','Fixed up to 128 valid windows per donor'],['Task','Same-day rainfall–runoff simulation, including target-day meteorology']])
  for name in ['pipeline_status.json','status.json','full_record_coverage_summary.json']:
   s=read(name)
   if s:out+='<h3>'+name+'</h3>'+table(['Field','Value'],list(s.items()))
  return out+'<p>Coverage varies by catchment. Missing observations are not filled. The requested period ends in early January 2023, not December 2023. The original catchment IDs and climate labels remain fixed.</p>'
 if section=='features':return '<h2>Daily ERA5 inputs</h2>'+table(['Input','Derivation'],list(c['feature_mapping'].items()))+'<p>Hourly fluxes are assigned by time bounds; instantaneous variables by timestamps. Complete 24-hour days are required. Daily feature and observation handling matches the earlier ERA5 experiment.</p>'
 if section=='coverage':
  data=rows('coverage_audit.csv')
  if not data:return '<p>Preparation is in progress. Completed catchments are cached for resume.</p>'
  out=[]
  for g in sorted({r.get('basin_group','unknown') for r in data}):
   rr=[r for r in data if r.get('basin_group','unknown')==g]; out.append([g,len(rr),sum(r['status']=='ok' for r in rr),sum(int(r['train_windows'])>0 for r in rr),sum(int(r['validation_windows'])>0 for r in rr),sum(int(r['test_windows'])>0 for r in rr)])
  return table(['Group','Catchments','Source audit pass','With train windows','With validation windows','With test observations/windows'],out)
 if section=='training':
  h=read('run_seed42/history.json',[])
  if not h:return '<p>Training begins after all source and protocol checks pass. See preparation.log and pipeline.log.</p>'
  return '<p>Each sampled epoch draws 256 valid windows per donor: 339,456 windows. The extended record supplies eligible training windows; an epoch is not an exhaustive pass through every historical day. Checkpoint selection uses donor validation only.</p>'+table(['Epoch','Training loss','Validation loss','Validation donors','Elapsed seconds'],[[r['epoch'],r['training_loss'],r['validation_loss'],r['validation_donors'],r['elapsed_seconds']] for r in h])
 if section in ['unseen','temporal']:
  directory='run_seed42' if section=='unseen' else 'run_seed42/temporal_test_2015_latest'; summaryname='unseen_evaluation_summary.json' if section=='unseen' else 'summary.json'; s=read(directory+'/'+summaryname)
  if not s:return '<p>This test has not finished. It is evaluated after checkpoint selection without fitting to test data.</p>'
  out=table(['Measure','Value'],[[k,v] for k,v in s.items() if k not in ['policy','checkpoint_sha256']]); group=rows(directory+('/unseen_group_metrics.csv' if section=='unseen' else '/group_metrics.csv'))
  out+='<h3>Results by group</h3>'+table(['Grouping','Category','Reserved','Scored','Median NSE','Median KGE'],[[r['grouping'],r['category'],r['basins'],r['scored_basins'],float(r['median_NSE']),float(r['median_KGE'])] for r in group])
  metrics=rows(directory+('/unseen_metrics.csv' if section=='unseen' else '/metrics.csv')); missing=[r for r in metrics if r['status']!='scored']
  if missing:out+='<h3>Catchments without a score</h3>'+table(['Basin','Reason','Observed test days'],[[r['basin_id'],r['status'],r['n']] for r in missing])
  out+='<h3>Catchment-level metrics</h3>'+table(['Basin','Status','Scored days','NSE','KGE','Volume bias [%]'],[[r['basin_id'],r['status'],r['n'],float(r['NSE']),float(r['KGE']),float(r['bias_percent'])] for r in metrics]); return out
 if section=='completion':
  d=read('completion.json')
  if not d:return '<p>Final verification is pending.</p>'
  out=table(['Final check','Value'],[['Phase',d['phase']],['Training','1950–2009'],['Validation','2010–2014'],['Testing','2015–2 January 2023, subject to basin coverage'],['Best epoch',d['best_epoch']],['Prior experiments',d['prior_experiments_preserved']]])
  out+='<h3>Final test performance</h3>'+table(['Test','Reserved IDs','Scored','Median NSE','Median KGE'],[[name,d[key]['requested_basins'],d[key]['scored_basins'],d[key]['median_NSE'],d[key]['median_KGE']] for name,key in [('Unseen catchments','unseen'),('Donor temporal test','temporal')]])
  return out+'<p>One unseen ID and 14 donor IDs lack valid forcing windows for the later test period; two more donors lack sufficient observations. All IDs remain assigned to their original split. Shared-polygon overlap limitations remain unchanged. These testing years differ from the earlier 2005–2009 baselines.</p>'
 raise ValueError(section)
if __name__=='__main__':print(json.dumps({'html':report(sys.argv[1])}))
