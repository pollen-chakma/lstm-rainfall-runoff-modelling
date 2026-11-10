from pathlib import Path
import csv, json, shutil, hashlib, os
import numpy as np

ROOT = Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
BASE = Path(__file__).resolve().parent
FEATURES = ['prcp','tmax','tmin','srad','vp','dayl','pet','q_obs_mm_day']

def prepare():
    cache = BASE/'cache'; cache.mkdir(exist_ok=True)
    comparison = list(csv.DictReader((ROOT/'LSTM_Performance_Comparison/basin_lstm_performance_comparison.csv').open()))
    assert len(comparison)==1426 and len({r['basin_id'] for r in comparison})==1426
    metadata = {r['basin_id']:{k:r[k] for k in ['basin_id','seen_unseen','basin_group','climate_class']} for r in comparison}
    manifest = dict(basins=list(metadata.values()),features=FEATURES,sources={},units=dict(prcp='mm/day',tmax='°C',tmin='°C',srad='W/m² (daylight mean)',vp='Pa',dayl='seconds',pet='mm/day',q_obs_mm_day='mm/day'),notes='ERA5 PET is native potential evaporation; Daymet PET uses the previous experiment derivation. These are observed forcing/discharge records, not LSTM predictions. Missing values remain missing. Daymet uses its existing model CSV (1980–2009); ERA5 uses the extended model daily preparation (1950–early 2023).')
    for source in ['Daymet','ERA5']:
        dest=BASE/(source.lower()+'_lumped_daily.csv'); folder=cache/source;folder.mkdir(exist_ok=True)
        if source=='Daymet':
            original=ROOT/'Model A Meteorology Only (Daymet)/daymet_forcing_discharge_1980_2009.csv'
            if not dest.exists(): shutil.copyfile(original,dest)
            assert dest.stat().st_size==original.stat().st_size
        elif not dest.exists():
            original=ROOT/'Model A Lumped (ERA5, 1950-2023)/prepared'
            tmp=dest.with_suffix('.csv.tmp')
            with tmp.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['basin_id','split','date']+FEATURES)
                for i,b in enumerate(metadata):
                    with np.load(original/(b+'.npz')) as z:
                        a=np.column_stack([z['x'],z['y']]);dates=z['dates'].astype(str)
                    assert a.shape==(len(dates),8)
                    split='training_donor' if metadata[b]['seen_unseen']=='Seen' else 'unseen_test'
                    w.writerows([b,split,d]+[format(float(x),'.9g') if np.isfinite(x) else '' for x in row] for d,row in zip(dates,a))
                    if (i+1)%200==0: print('ERA5 CSV',i+1,flush=True)
            tmp.replace(dest)
        # This cache is built exclusively from the delivered CSV, and can be regenerated.
        # Each basin is contiguous and dates strictly increase; no large dataframe is needed.
        rows=0; basins={}; current=None; dates=[]; values=[]; first=None;last=None
        def save():
            if current is None:return
            dd=np.asarray(dates,dtype='datetime64[D]').astype('int32'); vv=np.asarray(values,dtype='float64')
            assert len(np.unique(dd))==len(dd) and np.all(np.diff(dd)>0)
            assert current in metadata and vv.shape==(len(dd),8)
            np.save(folder/(current+'_dates.npy'),dd);np.save(folder/(current+'_values.npy'),vv)
            basins[current]=dict(rows=len(dd),start=dates[0],end=dates[-1],valid=[int(x) for x in np.isfinite(vv).sum(0)])
        with dest.open(newline='') as f:
            reader=csv.DictReader(f);assert reader.fieldnames==['basin_id','split','date']+FEATURES
            for row in reader:
                b=row['basin_id']
                if b!=current:
                    save();assert b not in basins;current=b;dates=[];values=[]
                expected='training_donor' if metadata[b]['seen_unseen']=='Seen' else 'unseen_test'
                assert row['split']==expected
                dates.append(row['date']);values.append([float(row[k]) if row[k] else np.nan for k in FEATURES]);rows+=1
            save()
        assert set(basins)==set(metadata)
        manifest['sources'][source]=dict(csv=dest.name,rows=rows,basin_count=len(basins),start=min(x['start'] for x in basins.values()),end=max(x['end'] for x in basins.values()),basins=basins,bytes=dest.stat().st_size)
        print('VERIFIED',source,rows,len(basins),flush=True)
        (BASE/'preparation_progress.json').write_text(json.dumps(dict(source=source,status='verified',rows=rows)))
    (BASE/'manifest.json').write_text(json.dumps(manifest))
    (BASE/'dataset_verification.json').write_text(json.dumps(dict(status='passed',csv_count=2,basins_per_source=1426,cache_built_from_csv=True,missing_values_preserved=True,records={k:{x:v[x] for x in ['rows','start','end','bytes']} for k,v in manifest['sources'].items()}),indent=2))
    print('PREPARATION COMPLETE',flush=True)

if __name__=='__main__': prepare()
