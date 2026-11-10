from pathlib import Path
import json,time,html
U=Path(__file__).resolve().parent
B=U.parent.parent/'Model A Lumped (Daymet + PCA10 Means Only)'
while True:
    try:
        s=json.loads((U/'status.json').read_text());phase=s['phase'];out=B/'outputs'
        if not out.exists():time.sleep(10);continue
        logfile=U/('preparation.log' if phase=='preparing' else 'two_file_input_check.log' if phase=='checking_two_file_inputs' else 'notebook_execution.log')
        lines=logfile.read_text().splitlines()[-5:] if logfile.exists() else []
        statusfile=out/'results_seed42/status.json'
        detail=json.loads(statusfile.read_text()) if statusfile.exists() else {}
        text='<html><head><meta charset="utf-8"><style>body{font:18px sans-serif;max-width:1100px;margin:35px}pre{background:#f3f5f7;padding:18px;white-space:pre-wrap}</style></head><body><h1>Daymet + PCA10 means only</h1><p>Seed 42. Same 1,326 donor and 100 unseen basins. Train 1980–1999; validate 2000–2004; test 2005–2009.</p><p>17 inputs: seven meteorological forcings and ten PCA means. No subbasin standard-deviation features. One LSTM layer, 64 hidden units, 20 epochs, 256 sampled windows per basin per epoch.</p><h2>Stage: '+html.escape(phase)+'</h2><pre>'+html.escape('\n'.join(lines))+'</pre><pre>'+html.escape(json.dumps(detail,indent=2))+'</pre><p>Refresh this page to see the latest saved progress.</p><p>Only required input files: Daymet_PCA10_Means_Only_LSTM.ipynb and daymet_pca10_means_only_discharge_1980_2009.csv.</p>'
        if phase=='complete':text+='<p><a href="model_report.html">Open the completed model report</a></p>'
        if phase=='error':text+='<pre>'+html.escape(s.get('message',''))+'</pre>'
        text+='</body></html>';tmp=out/'progress.tmp.html';tmp.write_text(text);tmp.replace(out/'progress.html')
        if phase in ['complete','error']:break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(15)
