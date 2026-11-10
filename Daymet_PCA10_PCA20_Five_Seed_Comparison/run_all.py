from pathlib import Path
import json,os,sys,subprocess,time,fcntl
from jupyter_client import KernelManager
B=Path(__file__).resolve().parent
def atomic(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def execute(path):
    nb=json.loads(path.read_text())
    def save():
        q=path.with_suffix('.tmp.ipynb');q.write_text(json.dumps(nb));q.replace(path)
    kernel=nb.get('metadata',{}).get('kernelspec',{}).get('name','model_a_torch')
    km=KernelManager(kernel_name=kernel);km.start_kernel(cwd=str(path.parent))
    kc=km.client();kc.start_channels();kc.wait_for_ready(timeout=180)
    try:
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            print('CELL START',i,flush=True);c['outputs']=[];failed=False
            mid=kc.execute(''.join(c['source']),store_history=True)
            while True:
                m=kc.get_iopub_msg(timeout=7200)
                if m['parent_header'].get('msg_id')!=mid:continue
                typ=m['header']['msg_type'];v=m['content']
                if typ=='status' and v['execution_state']=='idle':break
                if typ=='execute_input':c['execution_count']=v['execution_count']
                elif typ=='stream':c['outputs'].append(dict(output_type='stream',name=v['name'],text=v['text']));print(v['text'],end='',flush=True)
                elif typ in ('display_data','execute_result'):
                    o=dict(output_type=typ,data=v['data'],metadata=v.get('metadata',{}))
                    if typ=='execute_result':o['execution_count']=v['execution_count']
                    c['outputs'].append(o)
                elif typ=='error':c['outputs'].append(dict(output_type='error',ename=v['ename'],evalue=v['evalue'],traceback=v['traceback']));print('\n'.join(v['traceback']),flush=True);failed=True
            save();print('CELL FINISHED',i,flush=True)
            if failed:raise RuntimeError(f'Cell {i} failed')
        print('NOTEBOOK EXECUTION COMPLETE',flush=True)
    finally:save();kc.stop_channels();km.shutdown_kernel(now=True)
if len(sys.argv)>1:
    execute(Path(sys.argv[1]));sys.exit()
lock=(B/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.write(str(os.getpid()));lock.flush()
manifest=json.loads((B/'run_manifest.json').read_text());completed=[]
try:
    # Pair the same seed across setups consecutively. One process uses the GPU.
    for r in sorted(manifest,key=lambda r:(r['seed'],r['pcs'])):
        if r['reused']:completed.append(f"PCA{r['pcs']}/seed_{r['seed']}");continue
        d=Path(r['directory']);key=f"PCA{r['pcs']}/seed_{r['seed']}"
        if (d/'completion.json').exists() and json.loads((d/'completion.json').read_text()).get('status')=='complete':
            completed.append(key);continue
        atomic(B/'suite_status.json',dict(phase='running',active=key,completed=completed,job=os.environ.get('SLURM_JOB_ID'),updated=time.time()))
        print('RUN START',key,flush=True)
        with (d/'notebook_execution.log').open('a',buffering=1) as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),r['notebook']],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{key} failed; inspect its notebook_execution.log')
        assert (d/'completion.json').exists();completed.append(key);print('RUN COMPLETE',key,flush=True)
    atomic(B/'suite_status.json',dict(phase='analysing',completed=completed,updated=time.time()))
    analysis_python=B.parent/'Poor_Basin_Performance_Analysis/.venv/bin/python'
    subprocess.run([str(analysis_python),str(B/'analyse_multiseed.py')],check=True)
    atomic(B/'suite_status.json',dict(phase='complete',completed=completed,updated=time.time()))
    print('FIVE SEED EXPERIMENT COMPLETE',flush=True)
except Exception as e:
    atomic(B/'suite_status.json',dict(phase='error',error=str(e),completed=completed,updated=time.time()));raise
