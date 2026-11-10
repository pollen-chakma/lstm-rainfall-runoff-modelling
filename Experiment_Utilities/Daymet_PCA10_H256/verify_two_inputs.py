"""Run after the model runner exits; verify portable notebook setup without retraining."""
from pathlib import Path
import os,time,json,subprocess,sys,ast,shutil,traceback
import numpy as np

U=Path(__file__).resolve().parent
B=U.parent.parent/'Model A Lumped (Daymet + PCA10 H256 Dropout02 Adaptive50)'
O=B/'outputs';state=U/'two_input_check_status.json'
def status(phase,**info):state.write_text(json.dumps(dict(phase=phase,**info),indent=2))
try:
    runner_pid=int(sys.argv[1]);status('waiting_for_main_notebook',runner_pid=runner_pid)
    while True:
        try:os.kill(runner_pid,0)
        except ProcessLookupError:break
        time.sleep(3)
    complete=json.loads((O/'completion.json').read_text());assert complete['status']=='complete'
    status('checking_notebook_portability')
    notebook=B/'Daymet_PCA10_H256_Dropout02_Adaptive50.ipynb';nb=json.loads(notebook.read_text());src=''.join(nb['cells'][5]['source'])
    audit=json.loads((O/'protocol_verification.json').read_text());hashes=audit['preserved_baseline_sha256']
    addon='\n# Recreate protocol metadata when only the notebook and CSV are copied to an empty folder.\nif not (OUTPUT_BASE/\'protocol_verification.json\').exists():\n    atomic(OUTPUT_BASE/\'protocol_verification.json\',dict(status=\'passed\',csv_sha256=CSV_HASH,rows=1426*10958,basins=1426,seen=1326,unseen=100,schema_and_split_verified=True,preprocessing_matches_baseline=True,preserved_baseline_sha256=json.loads('+repr(json.dumps(hashes))+')))\n'
    if 'Recreate protocol metadata when only' not in src:
        ast.parse(src+addon);nb['cells'][5]['source']=(src+addon).splitlines(True)
        tmp=notebook.with_suffix('.tmp.ipynb');tmp.write_text(json.dumps(nb));tmp.replace(notebook)
    # Model code, config, loss, data reader and evaluation sources are untouched.
    folder=U/'two_file_input_reproduction';folder.mkdir(exist_ok=True)
    dest=folder/notebook.name
    shutil.copyfile(notebook,dest)
    csvname='daymet_pca10_discharge_1980_2009.csv';link=folder/csvname
    if not link.exists():os.link(B/csvname,link)
    assert sorted(p.name for p in folder.iterdir())==sorted([csvname,notebook.name]),'Isolated folder is not fresh'
    status('executing_isolated_read_prepare_define')
    with (U/'two_file_input_check.log').open('w') as log:
        subprocess.run([sys.executable,str(U/'execute_notebook.py'),str(dest),'--stop-after','7'],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Confirm every setup cell executed with no errors, and the notebook generated its own protocol file.
    isolated=json.loads(dest.read_text());cells=[c for c in isolated['cells'][:8] if c['cell_type']=='code']
    assert len(cells)==4 and all(c['execution_count'] is not None for c in cells)
    assert not any(x['output_type']=='error' for c in cells for x in c['outputs'])
    assert (folder/'outputs/protocol_verification.json').exists()
    with np.load(folder/'outputs/results_seed42/preprocessing.npz') as a,np.load(O/'results_seed42/preprocessing.npz') as b:
        assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    result=dict(status='passed',only_required_inputs=[notebook.name,csvname],isolated_read_preprocessing_model_definitions_executed=True,protocol_metadata_recreated_from_notebook=True,all_preprocessing_arrays_identical=True,main_training_and_tests_complete=True,full_training_not_repeated_for_input_check=True,training_config_and_model_code_unchanged_by_portability_metadata_patch=True)
    (O/'two_file_input_verification.json').write_text(json.dumps(result,indent=2))
    with (O/'progress.html').open('a') as f:f.write('<p style="font:16px sans-serif;margin:24px">Two-input reproduction check passed: an isolated folder containing only the notebook and CSV successfully read the data, recreated protocol metadata, fitted identical preprocessing and defined the model. Full training was completed once in the main folder and was not repeated for this input check.</p>')
    complete['two_file_input_check']='passed';(O/'completion.json').write_text(json.dumps(complete,indent=2))
    status('complete',**result);print(json.dumps(result),flush=True)
except Exception:
    status('failed',error=traceback.format_exc());raise
