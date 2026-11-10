from pathlib import Path
import json,sys,argparse,time
from jupyter_client import KernelManager
ap=argparse.ArgumentParser();ap.add_argument('notebook');ap.add_argument('--stop-after',type=int,default=None);args=ap.parse_args()
path=Path(args.notebook).resolve();nb=json.loads(path.read_text())
def save():
    temp=path.with_suffix('.tmp.ipynb');temp.write_text(json.dumps(nb));temp.replace(path)
last_save=time.monotonic()
km=KernelManager(kernel_name=nb['metadata']['kernelspec']['name']);km.start_kernel(cwd=str(path.parent))
kc=km.client();kc.start_channels();kc.wait_for_ready(timeout=180)
try:
    for i,cell in enumerate(nb['cells']):
        if args.stop_after is not None and i>args.stop_after:break
        if cell['cell_type']!='code':continue
        print('CELL START',i,flush=True);cell['outputs']=[];failed=False
        mid=kc.execute(''.join(cell['source']),store_history=True)
        while True:
            m=kc.get_iopub_msg(timeout=7200)
            if m['parent_header'].get('msg_id')!=mid:continue
            typ=m['header']['msg_type'];v=m['content']
            if typ=='status' and v['execution_state']=='idle':break
            if typ=='execute_input':cell['execution_count']=v['execution_count']
            elif typ=='stream':
                cell['outputs'].append(dict(output_type='stream',name=v['name'],text=v['text']));print(v['text'],end='',flush=True)
            elif typ in ('display_data','execute_result'):
                out=dict(output_type=typ,data=v['data'],metadata=v.get('metadata',{}))
                if typ=='execute_result':out['execution_count']=v['execution_count']
                cell['outputs'].append(out)
            elif typ=='error':
                cell['outputs'].append(dict(output_type='error',ename=v['ename'],evalue=v['evalue'],traceback=v['traceback']));print('\n'.join(v['traceback']),flush=True);failed=True
            if time.monotonic()-last_save>10:save();last_save=time.monotonic()
        save();print('CELL FINISHED',i,flush=True)
        if failed:raise RuntimeError(f'Notebook failed in cell {i}')
    print('NOTEBOOK EXECUTION COMPLETE',flush=True)
finally:save();kc.stop_channels();km.shutdown_kernel(now=True)
