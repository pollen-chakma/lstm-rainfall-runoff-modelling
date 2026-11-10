from pathlib import Path
import os,time,json,subprocess,sys,shutil,traceback
import numpy as np
U=Path(__file__).resolve().parent
B=U.parent.parent/'Model A Lumped (Daymet + PCA10 EA-LSTM H256 Dropout02 Adaptive50)'
O=B/'outputs';state=U/'two_input_check_status.json'
def status(phase,**info):state.write_text(json.dumps(dict(phase=phase,**info),indent=2))
try:
    pid=int(sys.argv[1]);status('waiting_for_main_notebook',runner_pid=pid)
    while True:
        try:os.kill(pid,0)
        except ProcessLookupError:break
        time.sleep(3)
    completed=json.loads((O/'completion.json').read_text());assert completed['status']=='complete'
    notebook=B/'Daymet_PCA10_EALSTM_H256_Dropout02_Adaptive50.ipynb'
    nb=json.loads(notebook.read_text());assert 'protocol_verification.json' in ''.join(nb['cells'][5]['source'])
    folder=U/'two_file_input_reproduction';folder.mkdir(exist_ok=True)
    dest=folder/notebook.name;shutil.copyfile(notebook,dest)
    name='daymet_pca10_discharge_1980_2009.csv';link=folder/name
    if not link.exists():os.link(B/name,link)
    assert sorted(p.name for p in folder.iterdir())==sorted([name,notebook.name]),'Isolated folder is not fresh'
    status('executing_isolated_read_prepare_define_and_gate_checks')
    with (U/'two_file_input_check.log').open('w') as log:
        subprocess.run([sys.executable,str(U/'execute_notebook.py'),str(dest),'--stop-after','7'],stdout=log,stderr=subprocess.STDOUT,check=True)
    isolated=json.loads(dest.read_text());cells=[c for c in isolated['cells'][:8] if c['cell_type']=='code']
    assert len(cells)==4 and all(c['execution_count'] is not None for c in cells)
    assert not any(x['output_type']=='error' for c in cells for x in c['outputs'])
    assert (folder/'outputs/protocol_verification.json').exists()
    assert json.loads((folder/'outputs/architecture_verification.json').read_text())['status']=='passed'
    with np.load(folder/'outputs/results_seed42/preprocessing.npz') as a,np.load(O/'results_seed42/preprocessing.npz') as b:
        assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    result=dict(status='passed',only_required_inputs=[notebook.name,name],isolated_read_preprocessing_model_definitions_and_gate_equivalence_checks_executed=True,protocol_metadata_recreated_from_notebook=True,all_preprocessing_arrays_identical=True,main_training_and_tests_complete=True,full_training_not_repeated_for_input_check=True)
    (O/'two_file_input_verification.json').write_text(json.dumps(result,indent=2))
    with (O/'progress.html').open('a') as f:f.write('<p style="font:16px sans-serif;margin:24px">Two-input check passed in an isolated folder containing only the notebook and CSV: loading, identical preprocessing, regenerated protocol metadata, model definitions and independent EA gate/gradient checks. Main training and testing completed once; full training was not repeated for this check.</p>')
    completed['two_file_input_check']='passed';(O/'completion.json').write_text(json.dumps(completed,indent=2))
    status('complete',**result);print(json.dumps(result),flush=True)
except Exception:
    status('failed',error=traceback.format_exc());raise
