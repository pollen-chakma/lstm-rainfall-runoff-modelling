from pathlib import Path
import json, time
import numpy as np
from dashboard import query, BASE, arrays

m=json.loads((BASE/'manifest.json').read_text());ids=[b['basin_id'] for b in m['basins']]
assert len(ids)==1426
cases=[]
for sources,basins,params,start,end in [(['Daymet','ERA5'],ids[:2],['prcp','q_obs_mm_day','tmax'],'2005-01-01','2009-12-31'),(['ERA5'],ids[:1],['vp'],'1950-01-01','1950-01-01'),(['Daymet'],ids[:1],['prcp'],'1950-01-01','1951-01-01'),(['Daymet','ERA5'],ids,['prcp','q_obs_mm_day','tmax'],'2005-01-01','2009-12-31')]:
    t=time.monotonic();r=query(dict(sources=sources,basins=basins,parameters=params,start=start,end=end))
    assert len(r['statistics'])==len(sources)*len(basins)*len(params)
    for row in r['statistics']:
        dd,vv=arrays(row['source'],row['basin']);a=vv[(dd>=np.datetime64(start).astype(int))&(dd<=np.datetime64(end).astype(int)),m['features'].index(row['parameter'])];valid=a[np.isfinite(a)]
        assert row['valid']==len(valid) and row['missing']==len(a)-len(valid)
        if len(valid):assert np.isclose(row['mean'],valid.mean()) and np.isclose(row['median'],np.median(valid))
        else:assert row['mean'] is None and row['median'] is None
    json.dumps(r,allow_nan=False)
    cases.append(dict(sources=sources,basins=len(basins),parameters=params,start=start,end=end,rows=len(r['statistics']),seconds=round(time.monotonic()-t,2)))
    print('PASS',cases[-1],flush=True)
# Compare the CSV-derived cache with independent source model records at representative dates.
for source,folder in [('ERA5','Model A Lumped (ERA5, 1950-2023)'),('Daymet','Model A Lumped (Daymet)')]:
    for b in [ids[0],ids[712],ids[-1]]:
        p=BASE.parent/folder/'prepared'/(b+'.npz')
        with np.load(p) as z:
            dates=z['dates'].astype('datetime64[D]').astype(int);values=np.column_stack([z['x'],z['y']])
        dd,vv=arrays(source,b)
        assert np.array_equal(dd,dates)
        assert np.allclose(vv,values,equal_nan=True,rtol=1e-8,atol=1e-7)
(BASE/'dashboard_verification.json').write_text(json.dumps(dict(status='passed',cases=cases,source_value_spotchecks=6),indent=2))
print('ALL DASHBOARD CHECKS PASSED',flush=True)
