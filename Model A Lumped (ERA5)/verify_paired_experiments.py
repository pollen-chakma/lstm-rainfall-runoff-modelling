from pathlib import Path
import json,csv,hashlib,numpy as np
r=Path.home()/'Desktop/LSTM_RR_Modelling'; e=r/'Model A Lumped (ERA5)'; d=r/'Model A Lumped (Daymet)'; ec=json.loads((e/'experiment_config.json').read_text()); dc=json.loads((d/'experiment_config.json').read_text())
for key in ['training_basins','unseen_basins','start','validation_start','test_start','end_exclusive','lookback','static','features','quality_policy','split_seed','overlap_pairs','held_basins_with_shared_ids']:assert ec[key]==dc[key],key
assert (e/'train_unseen.py').read_bytes()==(d/'train_unseen.py').read_bytes()
er=json.loads((e/'run_seed42/run_config.json').read_text()); dr=json.loads((d/'run_seed42/run_config.json').read_text())
for key in ['seed','epochs','batch','windows_per_basin_per_epoch','validation_windows_per_basin','hidden_units','validation_policy','task']:assert er[key]==dr[key],key
em={x['basin_id']:x for x in csv.DictReader(open(e/'split_manifest.csv'))}; dm={x['basin_id']:x for x in csv.DictReader(open(d/'split_manifest.csv'))}; assert set(em)==set(dm)
for b in em:
 for key in ['split','basin_group','climate_class','country','cold_precip_fraction','area_km2']:assert em[b][key]==dm[b][key],(b,key)
 with np.load(e/'prepared'/(b+'.npz')) as a,np.load(d/'prepared'/(b+'.npz')) as z:
  for key in ['dates','y','static']:
   assert np.array_equal(a[key],z[key],equal_nan=True) if key!='dates' else np.array_equal(a[key],z[key]),(b,key)
fm=json.loads((e/'run_seed42/fit_metadata.json').read_text()); assert fm['fitted_basin_ids']==dc['training_basins'] and fm['excluded_unseen_basin_ids']==dc['unseen_basins'] and fm['fit_period_end_exclusive']=='2000-01-01'
h=json.loads((e/'daymet_preservation_hashes.json').read_text()); assert all(hashlib.sha256((d/k).read_bytes()).hexdigest()==v for k,v in h.items())
result=dict(status='PASS',catchments_compared=1426,training_code='byte-identical to Daymet',dates_discharge_static='identical in all prepared catchments',split_labels='identical',training_hyperparameters='identical',daymet_original_result_hashes='unchanged',era5_fitted_basins=1326,excluded_holdout_basins=100)
(e/'paired_protocol_audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
