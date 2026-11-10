from pathlib import Path
import nbformat,traceback
from nbclient import NotebookClient
B=Path(__file__).resolve().parent; p=B/'Poor_Basin_Performance_Analysis.ipynb'
n=nbformat.read(p,as_version=4)
def start(cell,cell_index,**kw):print('CELL START',cell_index,flush=True)
c=NotebookClient(n,timeout=1800,kernel_name='basin_diagnostics',resources={'metadata':{'path':str(B)}},on_cell_start=start)
try:
 c.execute();print('NOTEBOOK EXECUTION COMPLETE',flush=True)
finally:
 tmp=p.with_suffix('.tmp.ipynb');nbformat.write(n,tmp);tmp.replace(p)
