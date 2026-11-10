"""Rebuild dashboard caches from just the two CSVs and retained metadata."""
from pathlib import Path
import csv,json
import numpy as np
BASE=Path(__file__).resolve().parent
m=json.loads((BASE/'manifest.json').read_text())
for source,meta in m['sources'].items():
    folder=BASE/'cache'/source;folder.mkdir(parents=True,exist_ok=True)
    current=None;dates=[];values=[];count=0;seen=set()
    def save():
        if current is None:return
        dd=np.asarray(dates,dtype='datetime64[D]').astype('int32');vv=np.asarray(values,dtype='float64')
        assert np.all(np.diff(dd)>0) and len(dd)==len(np.unique(dd))
        np.save(folder/(current+'_dates.npy'),dd);np.save(folder/(current+'_values.npy'),vv)
    for row in csv.DictReader((BASE/meta['csv']).open(newline='')):
        if row['basin_id']!=current:
            save();current=row['basin_id'];assert current not in seen;seen.add(current);dates=[];values=[]
        dates.append(row['date']);values.append([float(row[k]) if row[k] else np.nan for k in m['features']]);count+=1
    save();assert count==meta['rows'] and seen=={b['basin_id'] for b in m['basins']}
    print(source,count,'rows verified')
