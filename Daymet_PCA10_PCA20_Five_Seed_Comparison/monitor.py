from pathlib import Path
import time,json,html
B=Path(__file__).resolve().parent
while True:
    try:
        status=json.loads((B/'suite_status.json').read_text());manifest=json.loads((B/'run_manifest.json').read_text());rows=[]
        for r in sorted(manifest,key=lambda r:(r['seed'],r['pcs'])):
            d=Path(r['directory']);out=d/f"results_seed{r['seed']}";state='Reused completed baseline' if r['reused'] else 'Waiting'
            epoch='';val='';details=''
            if (out/'history.json').exists():
                history=json.loads((out/'history.json').read_text())
                if history:epoch=str(history[-1]['epoch'])+'/20';val=f"{history[-1]['validation_loss']:.4f}"
            if (out/'status.json').exists():
                s=json.loads((out/'status.json').read_text());state=s.get('phase',state)
                if state=='training':epoch=str(s['epoch'])+'/20';details=f"Batch {s['batch']}/{s['batches']}"
                if state=='testing':details=f"{s['test']}: {s['completed']}/{s['total']}"
            if (d/'completion.json').exists():state='Complete (reused)' if r['reused'] else 'Complete'
            key=f"PCA{r['pcs']}/seed_{r['seed']}"
            if state=='Waiting' and status.get('active')==key and status['phase']=='running':
                state='Loading/preparing CSV'
                log=d/'notebook_execution.log'
                if log.exists():
                    loaded=[v for v in log.read_text().splitlines() if 'CSV catchments loaded:' in v]
                    if loaded:details=loaded[-1]
            rows.append([f"PCA{r['pcs']}",r['seed'],state,epoch,val,details])
        text='<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="20"><style>body{font:17px sans-serif;margin:35px;max-width:1200px}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #ddd;text-align:left}th{background:#eef2f6}</style></head><body><h1>Daymet PCA10 versus PCA20: five-seed experiment</h1><p>H100 10GB GPU. Job 26058653. Seeds 42–46. Same 1,326 seen and 100 unseen basins. Training 1980–1999, validation 2000–2004, testing 2005–2009.</p><p>Suite phase: <b>'+html.escape(status['phase'])+'</b>. Completed new runs: '+str(sum(not r['reused'] and (Path(r['directory'])/'completion.json').exists() for r in manifest))+'/8. Epoch budget: 20 per run. LSTM hidden size: 64.</p><table><tr><th>Setup</th><th>Seed</th><th>Status</th><th>Epoch</th><th>Latest validation loss</th><th>Progress</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table><p>Outputs and notebooks: Desktop/LSTM_RR_Modelling/Daymet_PCA10_PCA20_Five_Seed_Comparison</p><p>Final analysis recalculates five-model ensemble metrics from averaged daily predictions and compares repeatability, basin-level consistency and low-flow/intermittent/dry subgroups.</p>'
        if status['phase']=='complete':text+='<p><a href="analysis_outputs/five_seed_report.html">Open completed five-seed report</a></p>'
        if status['phase']=='error':text+='<p>'+html.escape(status.get('error',''))+'</p>'
        text+='</body></html>'
        tmp=B/'progress.tmp.html';tmp.write_text(text);tmp.replace(B/'progress.html')
        if status['phase'] in ['complete','error']:break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(20)
