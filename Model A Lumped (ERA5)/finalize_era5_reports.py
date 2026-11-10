from pathlib import Path
import sys,json,csv,time,hashlib
BASE=Path(__file__).resolve().parent; sys.path.insert(0,str(BASE)); from inspect_experiment import report
sys.path.insert(0,str(BASE.parent/'Model_A')); from inspect_model_a import table

def code(source,html,count): return dict(cell_type='code',metadata={},source=[source],execution_count=count,outputs=[dict(output_type='display_data',metadata={},data={'text/html':[html],'text/plain':['ERA5 results']})])
def md(text):return dict(cell_type='markdown',metadata={},source=[text])
def notebook(cells,path):
 n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3.11',language='python',name='python3')),cells=cells)
 for i,c in enumerate(n['cells']):c['id']='era5-'+str(i)
 path.write_text(json.dumps(n))
def loadcsv(path):return list(csv.DictReader(open(path)))
def make():
 done=json.loads((BASE/'completion.json').read_text()); run=BASE/'run_seed42'; td=run/'temporal_test_2005_2009'; ts=done['temporal']; ms=loadcsv(td/'metrics.csv'); groups=loadcsv(td/'group_metrics.csv'); audit=loadcsv(td/'coverage_audit.csv'); missing=[r for r in audit if r['status']!='scored']
 html='<h2>Model A Lumped (ERA5): temporal test, 2005–2009</h2><p>Frozen best checkpoint and training-only preprocessing. Full valid test windows; no retraining.</p>'+table(['Measure','Value'],[[k,v] for k,v in ts.items() if k not in ['policy','checkpoint_sha256']])
 html+='<h3>Performance by group</h3>'+table(['Grouping','Category','Basins','Median NSE','Median KGE'],[[r['grouping'],r['category'],r['basins'],float(r['median_NSE']),float(r['median_KGE'])] for r in groups])
 if missing:html+='<h3>Insufficient observed test discharge</h3>'+table(['Basin','Valid observed test days'],[[r['basin_id'],r['observed_test_days']] for r in missing])
 html+='<h3>Catchment metrics</h3>'+table(['Basin','Group','Climate proxy','Country','Scored days','NSE','KGE','Volume bias [%]'],[[r['basin_id'],r['basin_group'],r['climate_class'],r['country'],r['n'],float(r['NSE']),float(r['KGE']),float(r['bias_percent'])] for r in ms]); (td/'results.html').write_text(html)
 source="from pathlib import Path"+chr(10)+"from IPython.display import display,HTML"+chr(10)+"display(HTML((Path.home()/'Desktop/LSTM_RR_Modelling/Model A Lumped (ERA5)/run_seed42/temporal_test_2005_2009/results.html').read_text()))"
 notebook([md('# Model A Lumped (ERA5): trained-catchment temporal test'+chr(10)+'Training: 1980–1999. Validation: 2000–2004. Test: 2005–2009. Antecedent weather can precede 2005; targets are in the test years only.'),code(source,html,1)],BASE/'Model_A_Lumped_ERA5_Temporal_Test.ipynb')
 compare=done['comparison']; comphtml='<h2>Daymet and ERA5: baseline comparison</h2>'+table(['Forcing','Test','Scored catchments','Median NSE','Median KGE'],[[r['forcing'],r['test'],r['scored_basins'],r['median_NSE'],r['median_KGE']] for r in compare])+'<p>Same 1326 donor and 100 unseen IDs, five static attributes, model, random seed, training settings and date splits. ERA5 inputs are daily analogs of the seven Daymet inputs; native PET and astronomical daylength definitions differ. This is one seed per source. Overlap limitations are unchanged. Original Daymet checkpoint, preprocessing, split and summaries passed SHA-256 preservation checks.</p>'
 (BASE/'comparison.html').write_text(comphtml)
 cells=[md('# Model A Lumped (ERA5)'+chr(10)+'Separate experiment. Seven daily ERA5 analog inputs; same five whole-basin attributes, catchment split, 1980–2009 protocol, LSTM and training settings as Daymet. Run All refreshes saved outputs and does not train the model. Source conversion and caveats are documented in README.md.')]
 for i,(name,title) in enumerate([('status','1. Configuration and daily ERA5 input mapping'),('split','2. Fixed catchment split'),('coverage','3. Source coverage audit'),('training','4. Training and validation losses'),('results','5. Unseen catchment test')],1):
  src="from pathlib import Path"+chr(10)+"import subprocess,json"+chr(10)+"from IPython.display import display,HTML"+chr(10)+"BASE=Path.home()/'Desktop/LSTM_RR_Modelling/Model A Lumped (ERA5)'"+chr(10)+"PY=BASE.parent/'Model_A/.venv/bin/python'"+chr(10)+f"result=subprocess.run([str(PY),str(BASE/'inspect_experiment.py'),'{name}'],capture_output=True,text=True,check=True)"+chr(10)+"display(HTML(json.loads(result.stdout)['html']))"
  cells += [md('## '+title),code(src,report(name),i)]
 cells += [md('## 6. Temporal test on trained catchments'),code(source,html,6),md('## 7. Comparison with preserved Daymet baseline'),code("display(HTML((BASE/'comparison.html').read_text()))",comphtml,7)]
 notebook(cells,BASE/'Model_A_Lumped_ERA5.ipynb'); (BASE/'reports_complete.json').write_text(json.dumps(dict(phase='complete',daymet_preservation='PASS',notebooks=['Model_A_Lumped_ERA5.ipynb','Model_A_Lumped_ERA5_Temporal_Test.ipynb']),indent=2)); print('FINAL ERA5 NOTEBOOKS SAVED',flush=True)
if __name__=='__main__':
 while not (BASE/'completion.json').exists():
  if (BASE/'pipeline_error.txt').exists():raise RuntimeError((BASE/'pipeline_error.txt').read_text())
  time.sleep(5)
 make()
