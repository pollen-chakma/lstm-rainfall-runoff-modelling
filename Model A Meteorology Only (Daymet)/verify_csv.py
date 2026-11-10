from pathlib import Path
import json,csv,hashlib,time
import numpy as np
from itertools import groupby
B=Path(__file__).resolve().parent;S=B.parent/'Model A Lumped (Daymet)'
start=time.time();rows=0;basins=0
with open(B/'daymet_forcing_discharge_1980_2009.csv',newline='') as f:
 r=csv.reader(f);header=next(r)
 assert header==['basin_id','split','date','prcp','tmax','tmin','srad','vp','dayl','pet','q_obs_mm_day']
 for bid,g in groupby(r,key=lambda a:a[0]):
  a=list(g);v=np.asarray([x[3:] for x in a],dtype='float32');dates=np.array([x[2] for x in a],dtype='datetime64[D]')
  with np.load(S/'prepared'/(bid+'.npz')) as z:
   assert np.array_equal(v[:,:7],z['x'],equal_nan=True),bid
   assert np.array_equal(v[:,7],z['y'],equal_nan=True),bid
   assert np.array_equal(dates,z['dates'].astype('datetime64[D]')),bid
  basins+=1;rows+=len(a)
  if basins%200==0:print('VERIFIED',basins,flush=True)
assert basins==1426 and rows==15626108
result=dict(passed=True,basins=basins,rows=rows,all_meteorology_values_exact_float32=True,all_discharge_values_exact_float32=True,all_dates_exact=True,attributes_in_csv=0,seconds=round(time.time()-start))
(B/'csv_roundtrip_verification.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
