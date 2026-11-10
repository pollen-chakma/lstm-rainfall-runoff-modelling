from pathlib import Path
import sys,csv,json,time,hashlib,collections,concurrent.futures,traceback
from decimal import Decimal
import numpy as np
from netCDF4 import Dataset
BASE=Path(__file__).resolve().parent
PILOT=BASE.parent/'Model_A'
sys.path.insert(0,str(PILOT))
from prepare_daymet import numeric
from fast_dates import dates
ROOT=Path('/project/6102174/data/misc-data/CAMELS-SPAT')
FEATURES=['prcp','tmax','tmin','srad','vp','dayl','pet']
STATIC=['basin_area','elev_mean','slope_mean','stream_density','elongation_ratio']
START,VAL,TEST,END='1980-01-01','2000-01-01','2005-01-01','2010-01-01'
L=365

def atomic(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)
def writecsv(path,rows):
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def task(item):
 row,attrs=item; basin=row['Country']+'_'+row['Station_id']; group=row['subset_category']
 cache=BASE/'prepared'/(basin+'.npz')
 try:
  area=float(attrs[0]); assert np.isfinite(area) and area>0,'Invalid drainage area'
  fp=ROOT/'forcing'/group/'daymet/daymet-lumped'/(basin+'_daymet_lumped.nc')
  op=ROOT/'observations'/group/'obs-daily'/(basin+'_daily_flow_observations.nc')
  d=np.arange(np.datetime64(START),np.datetime64(END))
  cache_ready=False
  if cache.exists():
   try:
    with np.load(cache) as z:
     xx=z['x']; y=z['y']; assert np.array_equal(z['dates'],d.astype(str)) and xx.shape==(len(d),7) and len(y)==len(d)
    cache_ready=True
   except Exception: cache_ready=False
  if not cache_ready:
   with Dataset(fp) as ds:
    fd=dates(ds); assert len(ds.dimensions['hru'])==1
    assert ds['prcp'].units=='mm/day' and ds['pet'].units=='mm/day'
    assert ds['tmax'].units=='degrees C' and ds['tmin'].units=='degrees C'
    x=np.column_stack([numeric(ds[v]).reshape(len(fd),-1)[:,0] for v in FEATURES])
   with Dataset(op) as ds:
    od=dates(ds,True); assert ds['q_obs'].units=='m3 s-1'
    q=numeric(ds['q_obs'])
    if 'q_obs_is_partial_day' in ds.variables: q[numeric(ds['q_obs_is_partial_day'])==1]=np.nan
   d=np.arange(np.datetime64(START),np.datetime64(END)); xx=np.full((len(d),7),np.nan,dtype='float32'); y=np.full(len(d),np.nan,dtype='float32')
   ii=np.searchsorted(d,fd); ok=(ii<len(d))&(fd>=d[0]); xx[ii[ok]]=x[ok]
   ii=np.searchsorted(d,od); ok=(ii<len(d))&(od>=d[0]); y[ii[ok]]=q[ok]*86.4/area; y[y<0]=np.nan
  bad=np.r_[0,np.cumsum(~np.isfinite(xx).all(1))]; good=np.zeros(len(d),bool); good[L-1:]=(bad[L:]-bad[:-L])==0
  masks=[d<np.datetime64(VAL),(d>=np.datetime64(VAL))&(d<np.datetime64(TEST)),d>=np.datetime64(TEST)]
  counts=[int((m&good&np.isfinite(y)).sum()) for m in masks]
  k=masks[0]&np.isfinite(xx[:,:3]).all(1)&(xx[:,0]>=0)
  total=float(np.sum(xx[k,0],dtype='float64'))
  frac=float(np.sum(xx[k&((xx[:,1]+xx[:,2])/2<0),0],dtype='float64')/total) if total>0 else np.nan
  climate='rain_proxy' if frac<.1 else 'mixed_proxy' if frac<.4 else 'snow_proxy' if np.isfinite(frac) else 'unknown'
  if not cache_ready: np.savez_compressed(cache,dates=d.astype(str),x=xx,y=y,static=np.array(attrs,dtype='float64'))
  return dict(basin_id=basin,country=row['Country'],basin_group=group,climate_class=climate,cold_precip_fraction=frac,train_windows=counts[0],validation_windows=counts[1],test_windows=counts[2],area_km2=area,status='ok',error='')
 except Exception as e:
  return dict(basin_id=basin,country=row['Country'],basin_group=group,climate_class='unknown',cold_precip_fraction=np.nan,train_windows=0,validation_windows=0,test_windows=0,area_km2=attrs[0],status='failed',error=repr(e))

def allocate(groups,n):
 keys=sorted(groups); assert len(keys)<=n
 counts={k:1 for k in keys}; left=n-len(keys)
 # Proportional allocation after ensuring representation of every nonempty stratum.
 while left:
  available=[k for k in keys if counts[k]<len(groups[k])]
  assert available,'Insufficient eligible holdout catchments'
  k=max(available,key=lambda k:((n*len(groups[k])/sum(len(groups[x]) for x in keys))-counts[k],len(groups[k]),k))
  counts[k]+=1; left-=1
 return counts

def main():
 BASE.mkdir(exist_ok=True); (BASE/'prepared').mkdir(exist_ok=True)
 meta=list(csv.DictReader(open(ROOT/'camels-spat-metadata.csv'))); assert len(meta)==1426
 with open(ROOT/'attributes/attributes-lumped.csv') as f:
  reader=csv.reader(f); header=next(reader); ar=list(reader)
 selected=[]
 for name in STATIC:
  matches=[r for r in ar if r[1]==name]; assert len(matches)==1,(name,len(matches)); selected.append(matches[0])
 items=[]
 for row in sorted(meta,key=lambda r:r['Country']+'_'+r['Station_id']):
  bi=header.index(row['Country']+'_'+row['Station_id']); vals=[]
  for rr in selected:
   try: vals.append(float(rr[bi]))
   except ValueError: vals.append(np.nan)
  items.append((row,vals))
 atomic(BASE/'status.json',dict(phase='preparing',completed=0,total=1426))
 rows=[]; started=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  for row in pool.map(task,items,chunksize=1):
   rows.append(row)
   if len(rows)%20==0 or len(rows)==1426:
    atomic(BASE/'status.json',dict(phase='preparing',completed=len(rows),total=1426,failed=sum(r['status']!='ok' for r in rows),elapsed_seconds=round(time.time()-started)))
    print('PREPARED',len(rows),'/1426; failures',sum(r['status']!='ok' for r in rows),flush=True)
 writecsv(BASE/'coverage_audit.csv',rows)
 failed=[r for r in rows if r['status']!='ok']; assert not failed,('Source-data errors; see coverage_audit.csv',failed[:3])
 # Catchments without training windows must be in holdout; never silently reduce 1326.
 forced=[r for r in rows if r['train_windows']==0]
 assert len(forced)<=100 and all(r['test_windows']>=365 for r in forced),('Exact split not feasible with these periods',forced)
 rng=np.random.default_rng(20261031)
 forcedids={r['basin_id'] for r in forced}
 groups=collections.defaultdict(list)
 for r in rows:
  if r['basin_id'] not in forcedids and r['test_windows']>=365:
   groups[(r['basin_group'],r['climate_class'],r['country'])].append(r['basin_id'])
 quotas=allocate(groups,100-len(forced)) if len(forced)<100 else {}
 held=set(forcedids)
 for key in sorted(quotas): held.update(rng.choice(sorted(groups[key]),size=quotas[key],replace=False).tolist())
 assert len(held)==100
 train=[r['basin_id'] for r in rows if r['basin_id'] not in held]; assert len(train)==1326
 assert all(r['train_windows']>0 for r in rows if r['basin_id'] in train)
 assert sum(r['validation_windows'] for r in rows if r['basin_id'] in train)>0
 for r in rows: r['split']='unseen_test' if r['basin_id'] in held else 'training_donor'
 writecsv(BASE/'split_manifest.csv',rows)
 (BASE/'training_basins.txt').write_text('\n'.join(train)+'\n'); (BASE/'unseen_test_basins.txt').write_text('\n'.join(sorted(held))+'\n')
 # Shared polygon IDs are a screening proxy, not a complete geometric overlap audit.
 idsets={}; byid=collections.defaultdict(set)
 for r in rows:
  p=ROOT/'attributes'/r['basin_group']/(r['basin_id']+'_attributes.csv')
  with open(p) as f: ids=next(csv.reader(f))[4:]
  ids={str(Decimal(s).normalize()) for s in ids}; idsets[r['basin_id']]=ids
  if r['basin_id'] in train:
   for pid in ids: byid[pid].add(r['basin_id'])
 overlaps=[]
 for b in sorted(held):
  matches=collections.Counter(t for pid in idsets[b] for t in byid[pid])
  overlaps.extend(dict(unseen_basin=b,training_basin=t,shared_polygon_ids=n) for t,n in sorted(matches.items()))
 if overlaps: writecsv(BASE/'shared_polygon_overlap_screen.csv',overlaps)
 else: (BASE/'shared_polygon_overlap_screen.csv').write_text('unseen_basin,training_basin,shared_polygon_ids\n')
 config=dict(dataset_root=str(ROOT),features=FEATURES,static=STATIC,start=START,validation_start=VAL,test_start=TEST,end_exclusive=END,lookback=L,training_basins=train,unseen_basins=sorted(held),split_seed=20261031,climate_definition='Pre-2000 Daymet precipitation fraction on days with (tmax+tmin)/2 < 0 C: rain_proxy <0.1; mixed_proxy 0.1–<0.4; snow_proxy >=0.4. Meteorology-only stratification, not formal climate zones.',quality_policy='Exclude negative and partial-day discharge; retain estimated/ice-affected values for comparability with pilot.',overlap_screen='Exact shared subbasin polygon IDs only; not exhaustive geometry. Preserve requested exact 1326/100 split.',overlap_pairs=len(overlaps),held_basins_with_shared_ids=len({r['unseen_basin'] for r in overlaps}),preprocessing_policy='All fitted model/scaling/imputation parameters use 1326 donors only. Heldout targets are never loaded by training process.')
 atomic(BASE/'experiment_config.json',config)
 strata=collections.Counter((r['split'],r['basin_group'],r['climate_class'],r['country']) for r in rows)
 writecsv(BASE/'stratum_counts.csv',[dict(split=k[0],basin_group=k[1],climate_class=k[2],country=k[3],count=v) for k,v in sorted(strata.items())])
 atomic(BASE/'status.json',dict(phase='prepared',training_basins=1326,unseen_basins=100,overlap_pairs=len(overlaps),elapsed_seconds=round(time.time()-started)))
 print('FIXED SPLIT READY: 1326 training donors / 100 unseen; shared-ID pairs',len(overlaps),flush=True)
if __name__=='__main__':
 try: main()
 except Exception:
  atomic(BASE/'status.json',dict(phase='failed',stage='preparation',error=traceback.format_exc())); raise
