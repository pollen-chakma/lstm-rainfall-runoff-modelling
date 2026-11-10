from pathlib import Path
import subprocess,sys,json,os,shutil,fcntl,hashlib
import numpy as np
U=Path(__file__).resolve().parent
ROOT=Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
B=ROOT/'Model A Lumped (Daymet + PCA10 Means Only)'
lock=(U/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock.write(str(os.getpid()));lock.flush()
def status(phase,**extra):
    p=U/'status.json';t=U/'status.tmp';t.write_text(json.dumps(dict(phase=phase,**extra),indent=2));t.replace(p)
try:
    status('preparing')
    with (U/'preparation.log').open('a') as f:subprocess.run([sys.executable,str(U/'prepare_experiment.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
    status('modelling')
    out=B/'outputs';notebook=B/'Daymet_PCA10_Means_Only_LSTM.ipynb'
    if not (out/'completion.json').exists():
        with (U/'notebook_execution.log').open('a') as f:subprocess.run([sys.executable,str(U/'execute_notebook.py'),str(notebook)],stdout=f,stderr=subprocess.STDOUT,check=True)
    assert json.loads((out/'completion.json').read_text())['status']=='complete'
    # An isolated two-input run verifies loading, preprocessing and model definitions.
    # Full training has already been performed using exactly those two input files.
    status('checking_two_file_inputs')
    proof=U/'two_file_input_check';proof.mkdir(exist_ok=True)
    csvname='daymet_pca10_means_only_discharge_1980_2009.csv'
    target=proof/csvname
    if not target.exists():
        try:os.link(B/csvname,target)
        except OSError:shutil.copy2(B/csvname,target)
    shutil.copy2(notebook,proof/notebook.name)
    with (U/'two_file_input_check.log').open('w') as f:subprocess.run([sys.executable,str(U/'execute_notebook.py'),str(proof/notebook.name),'--stop-after','7'],stdout=f,stderr=subprocess.STDOUT,check=True)
    main=np.load(out/'results_seed42/preprocessing.npz');isolated=np.load(proof/'outputs/results_seed42/preprocessing.npz')
    assert set(main.files)==set(isolated.files)
    for key in main.files:assert np.array_equal(main[key],isolated[key]),key
    a=json.loads((out/'results_seed42/data_audit.json').read_text());b=json.loads((proof/'outputs/results_seed42/data_audit.json').read_text())
    assert a['csv_sha256']==b['csv_sha256'] and a['rows']==b['rows']==1426*10958
    v=dict(status='passed',only_required_inputs=[notebook.name,csvname],isolated_loading_preprocessing_model_definitions_verified=True,all_scaler_arrays_identical=True,full_training_completed_once_in_main_folder=True,full_training_not_repeated_for_input_check=True)
    (out/'two_file_input_verification.json').write_text(json.dumps(v,indent=2))
    status('complete',output_directory=str(B));print('EXPERIMENT AND TWO-FILE INPUT CHECK COMPLETE',flush=True)
except Exception as e:
    status('error',message=str(e));raise
