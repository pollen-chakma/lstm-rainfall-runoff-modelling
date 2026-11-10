from pathlib import Path
import csv,json,collections
import numpy as np
b=Path(__file__).resolve().parent;r=b/'run_seed42'
assert json.loads((b/'status.json').read_text())['phase']=='complete'
metrics=list(csv.DictReader(open(r/'unseen_metrics.csv'))); c=json.loads((b/'experiment_config.json').read_text())
assert len(metrics)==100 and {m['basin_id'] for m in metrics}==set(c['unseen_basins'])
overlap={x['unseen_basin'] for x in csv.DictReader(open(b/'shared_polygon_overlap_screen.csv'))}
for row in metrics: row['shared_polygon_id_flag']=int(row['basin_id'] in overlap)
with open(r/'unseen_metrics_with_overlap.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
summary=[]
for label,subset in [('all_unseen',metrics),('no_shared_polygon_ids',[m for m in metrics if not m['shared_polygon_id_flag']]),('shared_polygon_ids',[m for m in metrics if m['shared_polygon_id_flag']])]:
 summary.append(dict(screen_group=label,catchments=len(subset),median_NSE=float(np.nanmedian([float(m['NSE']) for m in subset])) if subset else None,median_KGE=float(np.nanmedian([float(m['KGE']) for m in subset])) if subset else None))
with open(r/'overlap_screen_metrics.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('COMPLETE: 1326 trained, 100 unseen scored')
print('SUMMARY',json.loads((r/'summary.json').read_text()))
print('OVERLAP SCREEN',summary)
print('PREDICTIONS',len(list(r.glob('*_predictions.csv'))),'EPOCHS',len(json.loads((r/'history.json').read_text())))
