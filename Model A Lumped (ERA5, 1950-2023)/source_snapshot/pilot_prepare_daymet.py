"""Model A pilot: daily Daymet, lumped landscape attributes, outlet discharge."""
from pathlib import Path
import csv, json
import numpy as np
from netCDF4 import Dataset, num2date
ROOT = Path('/project/6102174/data/misc-data/CAMELS-SPAT')
OUT = Path(__file__).resolve().parent
FEATURES = ['prcp','tmax','tmin','srad','vp','dayl','pet']
STATIC = ['basin_area','elev_mean','slope_mean','stream_density','elongation_ratio']
# Fixed exploratory temporal split; not a final research protocol.
START, VAL, TEST, END = '1980-01-01','2000-01-01','2005-01-01','2010-01-01'
LOOKBACK = 365

def numeric(v):
    return np.ma.asarray(v[:], dtype=float).filled(np.nan)

def dates(ds, observation=False):
    t=ds['time']; b=ds['time_bnds']
    bounds=numeric(b)
    if b.dimensions[0] != 'time': bounds=bounds.T
    units=getattr(b,'units',t.units)
    starts=num2date(bounds[:,0],units,calendar=getattr(b,'calendar',getattr(t,'calendar','standard')))
    ends=num2date(bounds[:,1],units,calendar=getattr(b,'calendar',getattr(t,'calendar','standard')))
    assert all(abs((e-s).total_seconds()-86400)<1 for s,e in zip(starts,ends)), 'Non-daily bounds'
    assert all(s.hour==0 and s.minute==0 for s in starts), 'Bounds not midnight aligned'
    # Observation bounds lack units: their integer values use the time coordinate units.
    d=np.array([s.strftime('%Y-%m-%d') for s in starts],dtype='datetime64[D]')
    assert len(np.unique(d))==len(d), 'Duplicate dates'
    return d

def prepare():
    meta=list(csv.DictReader(open(ROOT/'camels-spat-metadata.csv')))
    with open(ROOT/'attributes/attributes-lumped.csv') as f:
        rd=csv.reader(f); header=next(rd); rows=list(rd)
    idx={name:header.index(name) for name in header[4:]}
    selected=[]
    for name in STATIC:
        matches=[r for r in rows if r[1]==name]
        assert len(matches)==1,(name,len(matches))
        selected.append(matches[0])
    groups=['headwater','meso-scale','macro-scale']; quotas={('CAN','headwater'):1,('CAN','meso-scale'):2,('CAN','macro-scale'):2,('USA','headwater'):2,('USA','meso-scale'):2,('USA','macro-scale'):1}
    counts=dict.fromkeys(quotas,0); records=[]; audit=[]
    cache=OUT/'prepared'; cache.mkdir(exist_ok=True)
    for row in sorted(meta,key=lambda r:r['Country']+'_'+r['Station_id']):
        group=row['subset_category']
        key=(row['Country'],group)
        if counts[key]>=quotas[key]: continue
        basin=row['Country']+'_'+row['Station_id']
        try:
            a=np.array([float(r[idx[basin]]) for r in selected])
            assert np.isfinite(a).all() and a[0]>0,'Invalid static attributes'
            fp=ROOT/'forcing'/group/'daymet/daymet-lumped'/(basin+'_daymet_lumped.nc')
            op=ROOT/'observations'/group/'obs-daily'/(basin+'_daily_flow_observations.nc')
            with Dataset(fp) as ds:
                fd=dates(ds)
                units={v:ds[v].units for v in FEATURES}
                assert units['prcp']=='mm/day' and units['pet']=='mm/day',units
                assert units['tmax']=='degrees C' and units['tmin']=='degrees C',units
                assert len(ds.dimensions['hru'])==1
                x=np.column_stack([numeric(ds[v]).reshape(len(fd),-1)[:,0] for v in FEATURES])
            with Dataset(op) as ds:
                od=dates(ds,True)
                assert ds['q_obs'].units=='m3 s-1'
                q=numeric(ds['q_obs'])
                # Keep estimated and ice-affected observations for this exploratory pilot;
                # exclude partial-day observations and retain all quality flags in source.
                if 'q_obs_is_partial_day' in ds.variables:
                    q[numeric(ds['q_obs_is_partial_day'])==1]=np.nan
            d=np.arange(np.datetime64(START),np.datetime64(END))
            xx=np.full((len(d),len(FEATURES)),np.nan); yy=np.full(len(d),np.nan)
            ii=np.searchsorted(d,fd); ok=(ii<len(d))&(fd>=d[0]); xx[ii[ok]]=x[ok]
            ii=np.searchsorted(d,od); ok=(ii<len(d))&(od>=d[0]); yy[ii[ok]]=q[ok]*86.4/a[0]
            yy[yy<0]=np.nan
            weather=np.isfinite(xx).all(1)
            bad=np.r_[0,np.cumsum(~weather)]
            valid=np.zeros(len(d),bool); valid[LOOKBACK-1:]=(bad[LOOKBACK:]-bad[:-LOOKBACK])==0
            ranges=[d<np.datetime64(VAL),(d>=np.datetime64(VAL))&(d<np.datetime64(TEST)),d>=np.datetime64(TEST)]
            n=[int((mask&valid&np.isfinite(yy)).sum()) for mask in ranges]
            assert min(n)>365,('Insufficient usable windows',n)
            train=yy[ranges[0]&np.isfinite(yy)]; assert np.std(train)>0
            np.savez_compressed(cache/(basin+'.npz'),dates=d.astype(str),x=xx,y=yy,static=a)
            records.append(dict(basin_id=basin,group=group,windows=n,area_km2=float(a[0])))
            counts[key]+=1
            audit.append(dict(basin_id=basin,status='selected',detail=str(n)))
            print('SELECTED',basin,group,n,flush=True)
        except (AssertionError,ValueError,OSError,KeyError) as e:
            audit.append(dict(basin_id=basin,status='excluded',detail=str(e)))
    assert counts==quotas,counts
    config=dict(forcing='daymet',features=FEATURES,static=STATIC,start=START,validation_start=VAL,test_start=TEST,end_exclusive=END,lookback=LOOKBACK,basins=records,quality_policy='Exclude partial-day and negative discharge; retain estimated/ice-affected values for exploratory pilot. No test scores used for selection.')
    (OUT/'pilot_config.json').write_text(json.dumps(config,indent=2))
    with open(OUT/'pilot_selection_audit.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['basin_id','status','detail']); w.writeheader(); w.writerows(audit)
    print('PREPARED',len(records),'basins at',cache)
if __name__=='__main__': prepare()
