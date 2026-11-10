"""Verification helper; not needed to repeat the notebook analysis."""
from pathlib import Path
import tempfile, shutil, json, hashlib
import nbformat
from nbclient import NotebookClient
import pandas as pd

B=Path(__file__).resolve().parent
name='Poor_Basin_Performance_Analysis.ipynb'
csvname='basin_diagnostic_dataset.csv'
baseline=B/'analysis_outputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=nbformat.read(B/name,as_version=4)
assert all(c.get('execution_count') is not None for c in source.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in source.cells for o in c.get('outputs',[]))
prep=json.loads((B/'preparation_verification.json').read_text())
assert sha(B/csvname)==prep['csv_sha256']
print('Original outputs and CSV checksum verified',flush=True)
with tempfile.TemporaryDirectory(prefix='basin_two_file_') as tmp:
    D=Path(tmp)
    shutil.copy2(B/name,D/name)
    shutil.copy2(B/csvname,D/csvname)
    assert {p.name for p in D.iterdir()}=={name,csvname}
    n=nbformat.read(D/name,as_version=4)
    for c in n.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    def start(cell,cell_index,**kw):print('REPRO CELL',cell_index,flush=True)
    NotebookClient(n,timeout=1800,kernel_name='basin_diagnostics',
                   resources={'metadata':{'path':str(D)}},on_cell_start=start).execute()
    expected=sorted(p.name for p in baseline.glob('*.csv'))+['FINDINGS.md','analysis_completion.json']
    for filename in expected:
        if filename.endswith('.csv'):
            a=pd.read_csv(baseline/filename);b=pd.read_csv(D/'analysis_outputs'/filename)
            pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-10,atol=1e-12)
        else:
            assert sha(baseline/filename)==sha(D/'analysis_outputs'/filename),filename
    assert all(c.get('execution_count') is not None for c in n.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in n.cells for o in c.get('outputs',[]))
    info=dict(status='passed',input_files=[name,csvname],csv_sha256=sha(B/csvname),
              code_cells=sum(c.cell_type=='code' for c in n.cells),
              matching_output_tables_and_reports=expected,numerical_tolerance=dict(rtol=1e-10,atol=1e-12),
              starting_directory_contained_only_the_two_files=True)
    (B/'two_file_reproduction_verification.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)
