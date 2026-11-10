from pathlib import Path
import json,csv,time,traceback,concurrent.futures
import numpy as np
from netCDF4 import Dataset,num2date
BASE=Path(__file__).resolve().parent
C=json.loads((BASE/'experiment_config.json').read_text())
DAY=Path(C['daymet_reference']); ROOT=Path(C['dataset_root']); D=np.arange(np.datetime64(C['start']),np.datetime64(C['end_exclusive'])); N=len(D)
def atomic(p,obj):
 tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)
def writecsv(p,rows):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def decode(raw,v):
 units=v.units; cal=getattr(v,'calendar','standard'); first=num2date(float(raw.flat[0]),units,calendar=cal); unit=units.split(' since ')[0].strip().lower(); factor={'hours':3600,'hour':3600,'seconds':1,'minutes':60,'days':86400}[unit]
 origin=np.datetime64(first.strftime('%Y-%m-%dT%H:%M:%S'),'s'); return origin+np.rint((raw-raw.flat[0])*factor).astype('timedelta64[s]')
def daylength(lat):
 j=(D-D.astype('datetime64[Y]')).astype(int)+1; delta=.409*np.sin(2*np.pi*j/365-1.39); lat=np.deg2rad(lat); omega=np.arccos(np.clip(-np.tan(lat)*np.tan(delta),-1,1)); return np.clip(86400*omega/np.pi,0,86400)

def aggregate(v,idx,mode):
 valid=np.isfinite(v); count=np.bincount(idx[valid],minlength=N)
 if mode in ['sum','mean']:
  out=np.bincount(idx[valid],weights=v[valid],minlength=N).astype(float)
  if mode=='mean': out/=24
 else:
  out=np.full(N,np.inf if mode=='min' else -np.inf); (np.minimum.at if mode=='min' else np.maximum.at)(out,idx[valid],v[valid])
 out[count!=24]=np.nan
 return out

def select(hours):
 mask=(hours>=D[0].astype('datetime64[s]'))&(hours<np.datetime64(C['end_exclusive'],'s')); ii=np.flatnonzero(mask); assert len(ii)>0
 sl=slice(int(ii[0]),int(ii[-1])+1); selected=hours[sl]; idx=(selected.astype('datetime64[D]')-D[0]).astype(int); assert np.all(np.diff(selected)==np.timedelta64(1,'h')); return sl,idx

def read(ds,k,sl):
 v=ds[k]; a=np.ma.filled(v[sl],np.nan).astype('float64').reshape(-1); return a

def task(row):
 b=row['basin_id']; p=ROOT/'forcing'/row['basin_group']/'era5/era5-lumped'/(b+'_era5_lumped.nc')
 try:
  with np.load(DAY/'prepared'/(b+'.npz')) as z: dates=z['dates']; y=z['y']; static=z['static']; assert np.array_equal(dates,D.astype(str))
  with Dataset(p) as ds:
   assert len(ds.dimensions['hru'])==1
   for k,u in {'mtpr':'kg m**-2 s**-1','mper':'kg m**-2 s**-1','t':'K','e':'kPa','msdwswrf':'W m**-2'}.items(): assert ds[k].units==u,(k,ds[k].units)
   times=decode(np.asarray(ds['time'][:],dtype=float),ds['time']); bounds=np.asarray(ds['time_bnds'][:],dtype=float)
   if ds['time_bnds'].dimensions[0]!='time': bounds=bounds.T
   bh=decode(bounds,ds['time_bnds']); assert np.all(bh[:,1]-bh[:,0]==np.timedelta64(1,'h')) and np.array_equal(times,bh[:,1])
   sf,fi=select(bh[:,0]); si,ii=select(times)
   prec=read(ds,'mtpr',sf); rad=read(ds,'msdwswrf',sf); pet=read(ds,'mper',sf); temp=read(ds,'t',si); vp=read(ds,'e',si)
   assert np.nanmin(prec)>-1e-6 and np.nanmin(rad)>-1 and np.nanmin(temp)>150 and np.nanmax(temp)<350 and np.nanmin(vp)>=0
   lat=float(np.ma.filled(ds['latitude'][0],np.nan).reshape(-1)[0]); dayl=daylength(lat)
   daily_prec=aggregate(np.maximum(prec,0)*3600,fi,'sum'); energy=aggregate(np.maximum(rad,0)*3600,fi,'sum'); daily_pet=-aggregate(pet*3600,fi,'sum')
   srad=np.divide(energy,dayl,out=np.zeros_like(energy),where=dayl>0); srad[~np.isfinite(energy)]=np.nan
   x=np.column_stack([daily_prec,aggregate(temp,ii,'max')-273.15,aggregate(temp,ii,'min')-273.15,srad,aggregate(vp,ii,'mean')*1000,dayl,np.maximum(daily_pet,0)]).astype('float32')
  assert np.all(x[np.isfinite(x[:,1])&np.isfinite(x[:,2]),1]>=x[np.isfinite(x[:,1])&np.isfinite(x[:,2]),2])
  good=np.zeros(N,bool); bad=np.r_[0,np.cumsum(~np.isfinite(x).all(1))]; L=C['lookback']; good[L-1:]=(bad[L:]-bad[:-L])==0
  masks=[D<np.datetime64(C['validation_start']),(D>=np.datetime64(C['validation_start']))&(D<np.datetime64(C['test_start'])),D>=np.datetime64(C['test_start'])]; counts=[int((m&good&np.isfinite(y)).sum()) for m in masks]
  np.savez_compressed(BASE/'prepared'/(b+'.npz'),dates=dates,x=x,y=y,static=static)
  return dict(basin_id=b,status='ok',train_windows=counts[0],validation_windows=counts[1],test_windows=counts[2],complete_forcing_days=int(np.isfinite(x).all(1).sum()),first_forcing_date=str(D[np.isfinite(x).all(1)][0]),last_forcing_date=str(D[np.isfinite(x).all(1)][-1]),negative_raw_daily_pet_days=int(np.sum(daily_pet<0)),source=str(p),error='')
 except Exception: return dict(basin_id=b,status='failed',train_windows=0,validation_windows=0,test_windows=0,complete_forcing_days=0,first_forcing_date='',last_forcing_date='',negative_raw_daily_pet_days=0,source=str(p),error=traceback.format_exc())

def main():
 # Synthetic checks: full-day sums/means, incomplete days rejected, extremes, unit conversion and polar day length.
 idx=np.repeat(np.arange(2),24); v=np.arange(48,dtype=float)
 assert np.allclose(aggregate(v,idx,'sum')[:2],[276,852]); assert np.allclose(aggregate(v,idx,'mean')[:2],[11.5,35.5]); assert np.allclose(aggregate(v,idx,'max')[:2],[23,47]); v[0]=np.nan; assert np.isnan(aggregate(v,idx,'sum')[0]); assert np.isclose(1e-5*3600*24,.864); assert np.all((daylength(69)>=0)&(daylength(69)<=86400)); print('AGGREGATION CHECKS PASS',flush=True)
 rows=list(csv.DictReader(open(DAY/'split_manifest.csv'))); assert len(rows)==1426; results=[]; start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
  for r in pool.map(task,rows,chunksize=1):
   results.append(r)
   if len(results)%25==0 or len(results)==1426:
    atomic(BASE/'status.json',dict(phase='preparing_era5',completed=len(results),total=1426,failed=sum(x['status']!='ok' for x in results),elapsed_seconds=round(time.time()-start))); print('PREPARED',len(results),'/1426; failures',sum(x['status']!='ok' for x in results),flush=True)
 writecsv(BASE/'coverage_audit.csv',results); failed=[r for r in results if r['status']!='ok']; assert not failed,[(r['basin_id'],r['error']) for r in failed[:3]]
 byid={r['basin_id']:r for r in results}; assert all(byid[b]['train_windows']>0 for b in C['training_basins']); assert all(byid[b]['test_windows']>=365 for b in C['unseen_basins'])
 for r in rows:
  for k in ['train_windows','validation_windows','test_windows']: r[k]=byid[r['basin_id']][k]
 writecsv(BASE/'split_manifest.csv',rows)
 assert (BASE/'training_basins.txt').read_bytes()==(DAY/'training_basins.txt').read_bytes() and (BASE/'unseen_test_basins.txt').read_bytes()==(DAY/'unseen_test_basins.txt').read_bytes()
 atomic(BASE/'status.json',dict(phase='prepared',training_basins=1326,unseen_basins=100,source='ERA5')); print('ERA5 PREPARATION PASS: fixed 1326/100 split',flush=True)
if __name__=='__main__':
 try: main()
 except Exception: atomic(BASE/'status.json',dict(phase='failed',error=traceback.format_exc())); raise
