from pathlib import Path
import csv,json,sys,html,subprocess,uuid
import numpy as np
BASE=Path(__file__).resolve().parent
RUN=BASE/'runs/20261031_212304'
CFG=json.loads((BASE/'pilot_config.json').read_text())
RC=json.loads((RUN/'run_config.json').read_text())
def table(headers,rows):
 def fmt(v):
  if isinstance(v,(float,np.floating)): return 'missing' if not np.isfinite(v) else f'{v:,.4f}'
  return str(v)
 head=''.join('<th style="padding:7px;text-align:left;border-bottom:2px solid #ccd">'+html.escape(str(x))+'</th>' for x in headers)
 body=''.join('<tr>'+''.join('<td style="padding:6px;border-bottom:1px solid #eee">'+html.escape(fmt(x))+'</td>' for x in row)+'</tr>' for row in rows)
 return '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:13px"><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'
def chart(title,series,ylabel,labels=None):
 W,H,L,R,T,B=1000,280,72,55,36,45
 finite=np.concatenate([np.asarray(y)[np.isfinite(y)] for name,y,color in series])
 lo=min(0,float(finite.min())); hi=float(finite.max()); hi=hi if hi>lo else lo+1; hi=lo+(hi-lo)*1.08
 n=max(len(y) for _,y,_ in series)
 def xx(i): return L+i/max(n-1,1)*(W-L-R)
 def yy(v): return H-B-(v-lo)/(hi-lo)*(H-T-B)
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}"><rect width="1000" height="280" fill="white"/>',f'<text x="{L}" y="20" font-family="sans-serif" font-size="16" font-weight="bold">{html.escape(title)}</text>']
 for v in np.linspace(lo,hi,5):
  y=yy(v); parts.extend([f'<line x1="{L}" y1="{y}" x2="{W-R}" y2="{y}" stroke="#e5e7eb"/>',f'<text x="{L-8}" y="{y+4}" text-anchor="end" font-size="11">{v:.2f}</text>'])
 for name,values,color in series:
  d=[]; active=False
  for i,v in enumerate(values):
   if not np.isfinite(v): active=False; continue
   d.append(('L' if active else 'M')+f'{xx(i):.2f},{yy(v):.2f}'); active=True
  parts.append(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="1.35"/>')
 for j,(name,_,color) in enumerate(series): parts.append(f'<text x="{W-250+j*120}" y="20" fill="{color}" font-size="12">{html.escape(name)}</text>')
 for i in np.linspace(0,n-1,5).astype(int):
  label=str(labels[i]) if labels is not None else str(i+1)
  parts.append(f'<text x="{xx(i)}" y="{H-21}" text-anchor="middle" font-size="11">{html.escape(label)}</text>')
 parts.append(f'<text transform="translate(15,{H/2}) rotate(-90)" text-anchor="middle" font-size="12">{html.escape(ylabel)}</text></svg>')
 return ''.join(parts)
def section(name):
 if name=='overview':
  rows=[]
  for b in CFG['basins']:
   with np.load(BASE/'prepared'/(b['basin_id']+'.npz')) as z: rows.append([b['basin_id'],b['group'],len(z['dates']),z['x'].shape[1],len(z['static']),*b['windows']])
  summary=table(['Setting','Value'],[['Run',RUN.name],['Device',RC.get('gpu_name','not recorded')],['Task','Same-day simulation; target-day weather included'],['Training','1980–1999'],['Validation','2000–2004'],['Testing','2005–2009'],['Lookback','365 days'],['LSTM','1 layer, 64 hidden units'],['Sampling','200 batches/epoch; batch 32; equal basin probability'],['Epochs / seed','20 / 42'],['Target','Discharge, mm/day']])
  return summary+'<h3>Prepared arrays and usable windows</h3>'+table(['Basin','Group','Calendar days','Weather variables','Static attributes','Train windows','Validation windows','Test windows'],rows)+'<p>Calendar days include missing records. Windows require complete weather; train/validation windows also require observed discharge. This deterministic ten-basin pilot is not a representative research sample.</p>'
 if name=='attributes':
  catalog=list(csv.DictReader(open(BASE/'audit_outputs/attribute_catalog.csv')))
  units={r['attribute']:r['unit'] for r in catalog if r['attribute'] in CFG['static']}
  rows=[]
  for b in CFG['basins']:
   with np.load(BASE/'prepared'/(b['basin_id']+'.npz')) as z: rows.append([b['basin_id'],*z['static']])
  return '<p>Actual unscaled whole-basin attributes used in this run. No PCA was applied. Soil and vegetation descriptors are not included in this initial topographic pilot.</p>'+table(['Basin']+[v+' ['+units.get(v,'')+']' for v in CFG['static']],rows)
 if name=='weather':
  units=['mm/day','degrees C','degrees C','W/m2','Pa','s','mm/day']
  desc=['Precipitation','Maximum temperature','Minimum temperature','Shortwave radiation','Vapour pressure','Day length','Potential evapotranspiration']
  out=table(['Variable','Meaning','Native input unit'],list(zip(CFG['features'],desc,units)))
  basin='CAN_01AF007'
  with np.load(BASE/'prepared'/(basin+'.npz')) as z:
   d,x,y=z['dates'],z['x'],z['y']; ix=np.flatnonzero((d>='1995-01-01')&(d<'1995-01-11'))
   rows=[[d[i],*x[i],y[i]] for i in ix]
   out+='<h3>Actual sample: '+basin+', 1–10 January 1995</h3>'+table(['Date']+[v+' ['+u+']' for v,u in zip(CFG['features'],units)]+['Observed Q [mm/day]'],rows)
   mask=(d>='1995-01-01')&(d<'1996-01-01')
   out+=chart('Daily precipitation — '+basin,[('Precipitation',x[mask,0],'#2563eb')],'mm/day',d[mask])
  return out+'<p>Observed discharge is shown for context; it is the target, not an input. Potential evapotranspiration is derived. Partial-day and negative observed flows were excluded; estimated and ice-affected flows were retained for this pilot.</p>'
 if name=='scaling':
  with np.load(RUN/'preprocessing.npz') as p:
   rows=[['Weather',n,float(m),float(s)] for n,m,s in zip(CFG['features'],p['xmean'],p['xstd'])]
   rows += [['Attribute',n,float(m),float(s)] for n,m,s in zip(CFG['static'],p['amean'],p['astd'])]
   rows += [['Target','Q [mm/day]',float(p['ymean']),float(p['ystd'])]]
  return '<p>Standardization: (raw value − training mean) / training standard deviation. Dynamic/target statistics use training-period records. Attribute statistics use the ten training catchments. This is temporal holdout, not unseen-basin evaluation.</p>'+table(['Type','Feature','Training mean','Training standard deviation'],rows)
 if name=='loss':
  h=json.loads((RUN/'history.json').read_text()); best=min(h,key=lambda x:x['validation_loss'])
  return '<p>Checkpoint selected by validation loss at epoch <b>'+str(best['epoch'])+'</b>. Loss is normalized discharge squared error, not NSE. Training uses sampled batches; validation uses all valid windows, so the two averages are not directly equivalent.</p>'+chart('Training and validation loss',[('Training',np.array([x['train_loss'] for x in h]),'#2563eb'),('Validation',np.array([x['validation_loss'] for x in h]),'#ea580c')],'Normalized loss')+table(['Epoch','Training loss','Validation loss'],[[x['epoch'],x['train_loss'],x['validation_loss']] for x in h])
 if name=='metrics':
  m=list(csv.DictReader(open(RUN/'metrics.csv')))
  rows=[[r['basin_id'],int(r['n']),float(r['NSE']),float(r['KGE']),float(r['bias_percent'])] for r in m]
  med=np.median([[r[2],r[3]] for r in rows],axis=0)
  return '<p><b>Median NSE: '+f'{med[0]:.3f}'+' · Median KGE: '+f'{med[1]:.3f}'+'</b>. Test: 2005–2009. Positive volume bias means overprediction. KGE uses correlation, variability ratio and mean ratio. Single-seed pilot results.</p>'+table(['Basin','Valid test days','NSE','KGE','Volume bias [%]'],rows)
 if name=='hydrographs':
  out='<p>Observed (dark) and predicted (orange) outlet discharge. Full test period and a 2005 close-up for each basin. Missing observations are gaps. Negative predictions remain visible. Each plot has its own vertical scale.</p>'
  for b in CFG['basins']:
   rs=list(csv.DictReader(open(RUN/(b['basin_id']+'_predictions.csv'))))
   d=np.array([r['date'] for r in rs]); y=np.array([float(r['observed_mm_day']) for r in rs]); q=np.array([float(r['predicted_mm_day']) for r in rs])
   out+=chart(b['basin_id']+' — full test period',[('Observed',y,'#1f2937'),('Predicted',q,'#ea580c')],'Discharge [mm/day]',d)
   k=(d>='2005-01-01')&(d<'2006-01-01')
   out+=chart(b['basin_id']+' — 2005 close-up',[('Observed',y[k],'#1f2937'),('Predicted',q[k],'#ea580c')],'Discharge [mm/day]',d[k])
  return out
 raise ValueError(name)
def build():
 dest=BASE/'Model_A_Results.ipynb'
 if dest.exists(): raise FileExistsError(dest)
 def cell(kind,text):
  c=dict(cell_type=kind,id=uuid.uuid4().hex[:8],metadata={},source=text.splitlines(True))
  if kind=='code': c.update(execution_count=None,outputs=[])
  return c
 cells=[cell('markdown','# Model A — data and results inspection\n\nExecuted inspection of the completed H100 pilot, run **20261031_212304**. Every displayed value comes from saved input arrays or run outputs. Running this notebook does not train a model.\n\nOriginal training notebook: Model_A_CAMELS_SPAT.ipynb.')]
 sections=[('overview','1. Dataset, configuration and split sizes'),('attributes','2. Actual whole-basin attributes'),('weather','3. Actual daily weather inputs and units'),('scaling','4. Training-only normalization'),('loss','5. Training and validation losses'),('metrics','6. Test performance by basin'),('hydrographs','7. Observed and predicted discharge — all ten basins')]
 for count,(name,title) in enumerate(sections,1):
  cells.append(cell('markdown','## '+title))
  source="from pathlib import Path\nimport subprocess, json\nfrom IPython.display import display, HTML\nBASE = Path.home() / 'Desktop/LSTM_RR_Modelling/Model_A'\nresult = subprocess.run([str(BASE/'.venv/bin/python'), str(BASE/'inspect_model_a.py'), '"+name+"'], capture_output=True, text=True, check=True)\ndisplay(HTML(json.loads(result.stdout)['html']))\n"
  c=cell('code',source); compile(source,'section_'+name,'exec')
  result=subprocess.run([sys.executable,str(Path(__file__)),name],capture_output=True,text=True,check=True)
  data=json.loads(result.stdout)
  c.update(execution_count=count,outputs=[dict(output_type='display_data',metadata={},data={'text/html':[data['html']],'text/plain':['Model A inspection: '+name]})])
  cells.append(c)
 cells.append(cell('markdown','## Interpretation\n\nThis pilot checks a lumped baseline with seven Daymet variables and five topographic attributes. It does not yet test sub-basin heterogeneity, unseen basins or multiple random seeds. Keep the test period untouched during tuning and use the same evaluation protocol when comparing models.'))
 nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3.11','language':'python','name':'python3'},'model_a_run':RUN.name},nbformat=4,nbformat_minor=5)
 dest.write_text(json.dumps(nb,indent=2)); print('CREATED',dest,'EXECUTED SECTIONS',len(sections),'BYTES',dest.stat().st_size)
if __name__=='__main__':
 if len(sys.argv)>1: print(json.dumps({'html':section(sys.argv[1])}))
 else: build()
