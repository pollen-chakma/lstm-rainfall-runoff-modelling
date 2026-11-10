from pathlib import Path
import json,subprocess,shutil,hashlib
import numpy as np
from PIL import Image
B=Path(__file__).resolve().parent
O=B/'analysis_outputs'
tables=['per_seed_group_performance.csv','repeatability_summary.csv','paired_seed_comparison.csv','basin_improvement_consistency.csv','ensemble_performance_summary.csv','hydrological_subgroup_comparison.csv','consistency_distribution.csv']
before={name:hashlib.sha256((O/name).read_bytes()).hexdigest() for name in tables}
path=B/'Five_Seed_Comparison_Analysis.ipynb'
nb=json.loads(path.read_text())
nb['cells'][0]['source']=('# Five-seed Daymet PCA10 versus PCA20 comparison\n\nThis notebook reads only `multiseed_basin_analysis_dataset.csv` beside it. It reproduces comparison tables and figures from verified per-seed scores, recalculated ensemble scores and observed hydrological diagnostics. The original training folders are not needed for this analysis. Use each seed notebook to rerun model training.\n').splitlines(keepends=True)
nb['cells'][1].update(source=(B/'report_analysis.py').read_text().splitlines(keepends=True),execution_count=None,outputs=[])
path.write_text(json.dumps(nb))
repro=B/'two_file_reproduction'
shutil.copy2(path,repro/path.name)
python=B.parent/'Model_A/.venv/bin/python'
for directory,logname in [(B,'analysis_notebook_execution.log'),(repro,'two_file_reproduction.log')]:
    with (B/logname).open('w') as log:
        subprocess.run([str(python),str(B/'run_all.py'),str(directory/path.name)],stdout=log,stderr=subprocess.STDOUT,check=True)
for name in tables:
    assert hashlib.sha256((O/name).read_bytes()).hexdigest()==before[name],name
    assert (O/name).read_bytes()==(repro/'analysis_outputs'/name).read_bytes(),name
figures=[f'{metric}_{kind}.png' for metric in ['NSE','KGE'] for kind in ['five_seed_repeatability','basin_consistency']]
for name in figures:
    with Image.open(O/name) as a,Image.open(repro/'analysis_outputs'/name) as b:assert np.array_equal(np.asarray(a),np.asarray(b)),name
v=dict(status='passed',input_files=[path.name,'multiseed_basin_analysis_dataset.csv'],tables_identical=tables,figures_pixel_identical=figures,numerical_tables_unchanged_by_layout=True,analysis_notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(O/'two_file_reproduction_verification.json').write_text(json.dumps(v,indent=2))
print('LAYOUT AND TWO-FILE REPRODUCTION PASSED; numerical tables unchanged.',flush=True)
