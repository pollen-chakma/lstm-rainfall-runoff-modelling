from pathlib import Path
import json,time
from jupyter_client import KernelManager
B=Path(__file__).resolve().parent
path=B/'Daymet_Meteorology_Only_LSTM.ipynb'
nb=json.loads(path.read_text())
def save():
 tmp=path.with_suffix('.tmp.ipynb');tmp.write_text(json.dumps(nb));tmp.replace(path)
km=KernelManager(kernel_name='model_a_torch')
km.start_kernel(cwd=str(B));kc=km.client();kc.start_channels();kc.wait_for_ready(timeout=180)
try:
 for i,cell in enumerate(nb['cells']):
  if cell['cell_type']!='code':continue
  print('CELL START',i,flush=True);cell['outputs']=[]
  msgid=kc.execute(''.join(cell['source']),store_history=True)
  failed=False
  while True:
   msg=kc.get_iopub_msg(timeout=7200)
   if msg['parent_header'].get('msg_id')!=msgid:continue
   typ=msg['header']['msg_type'];c=msg['content']
   if typ=='status' and c['execution_state']=='idle':break
   if typ=='execute_input':cell['execution_count']=c['execution_count']
   elif typ=='stream':
    cell['outputs'].append(dict(output_type='stream',name=c['name'],text=c['text']));print(c['text'],end='',flush=True)
   elif typ in ('display_data','execute_result'):
    out=dict(output_type=typ,data=c['data'],metadata=c.get('metadata',{}))
    if typ=='execute_result':out['execution_count']=c['execution_count']
    cell['outputs'].append(out)
   elif typ=='error':
    cell['outputs'].append(dict(output_type='error',ename=c['ename'],evalue=c['evalue'],traceback=c['traceback']));print(chr(10).join(c['traceback']),flush=True);failed=True
  save();print('CELL FINISHED',i,flush=True)
  if failed:raise RuntimeError('Notebook failed in cell '+str(i))
 print('NOTEBOOK EXECUTION COMPLETE',flush=True)
finally:
 save();kc.stop_channels();km.shutdown_kernel(now=True);del kc;del km
