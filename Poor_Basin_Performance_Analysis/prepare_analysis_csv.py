"""One-time extraction on Nibi. The delivered analysis notebook does not need this script."""
from pathlib import Path
import csv, json, hashlib, math
import numpy as np

R = Path('/home/pollen26/Desktop/LSTM_RR_Modelling')
B = R/'Poor_Basin_Performance_Analysis'
B.mkdir(exist_ok=True)
TARGET = B/'basin_diagnostic_dataset.csv'
assert not TARGET.exists(), 'Existing dataset: inspect before rebuilding.'
def read(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))
comparison = read(R/'LSTM_Performance_Comparison/basin_lstm_performance_comparison.csv')
assert len(comparison)==len({r['basin_id'] for r in comparison})==1426
pca = {r['basin_id']:r for r in read(R/'Subbasin_Heterogeneity_PCA/basin_pca10_mean_std_features.csv')}
assert set(pca)=={r['basin_id'] for r in comparison}
attr_path=Path('/project/6102174/data/misc-data/CAMELS-SPAT/attributes/attributes-lumped.csv')
selected = ['basin_area','elev_mean','slope_mean','stream_density','elongation_ratio',
            'open_water_area_total','reservoir_area_total','regolith_thickness_mean',
            'porosity_mean','log_permeability_mean','aridity1_mean','aridity2_mean',
            'fracsnow1_mean','fracsnow2_mean','forest_height_2020_mean',
            'lc2_evergreen_needleleaf_forest_fraction','lc2_evergreen_broadleaf_forest_fraction',
            'lc2_deciduous_needleleaf_forest_fraction','lc2_deciduous_broadleaf_forest_fraction',
            'lc2_mixed_forest_fraction','lc2_grassland_fraction','lc2_cropland_fraction',
            'lc2_permanent_wetland_fraction','lc2_urban_and_built_up_fraction',
            'clay_0-5cm_mean_mean','sand_0-5cm_mean_mean','silt_0-5cm_mean_mean']
attributes={a['Attribute']:a for a in read(attr_path) if a['Attribute'] in selected}
units={k:v['Unit'] for k,v in attributes.items()}
def num(x):
    try:
        v=float(x)
        return v if np.isfinite(v) and v not in (-999,-9999,-99999) else np.nan
    except (TypeError,ValueError): return np.nan
def safe_ratio(a,b): return float(a/b) if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan
def longest(mask):
    z=np.r_[False,np.asarray(mask,dtype=bool),False].astype(int)
    return int(np.max(np.flatnonzero(np.diff(z)==-1)-np.flatnonzero(np.diff(z)==1),initial=0))
def summaries(dates,x,y,start,end):
    dates=dates.astype('datetime64[D]'); mask=(dates>=np.datetime64(start))&(dates<np.datetime64(end))
    dt=dates[mask]; x=x[mask].astype(float); y=y[mask].astype(float)
    valid=np.isfinite(y)&(y>=0); q=y[valid]; o={}
    o['calendar_days']=int((np.datetime64(end)-np.datetime64(start))/np.timedelta64(1,'D'))
    o['available_days']=int(len(dt)); o['q_n']=int(len(q)); o['q_coverage']=len(q)/o['calendar_days']
    names=['q_mean_mm_day','q_median_mm_day','q_max_mm_day','q_std_mm_day','q_cv',
           'q05_mm_day','q10_mm_day','q90_mm_day','q95_mm_day','q99_mm_day',
           'zero_flow_fraction','near_zero_fraction','flashiness','q_seasonal_amplitude',
           'top1pct_flow_volume_fraction','q_longest_zero_run_days']
    o.update({k:np.nan for k in names})
    if len(q):
        qs=np.quantile(q,[.05,.10,.90,.95,.99]); mean=q.mean()
        o.update(q_mean_mm_day=float(mean),q_median_mm_day=float(np.median(q)),q_max_mm_day=float(q.max()),
                 q_std_mm_day=float(q.std()),q_cv=safe_ratio(q.std(),mean),
                 zero_flow_fraction=float(np.mean(q==0)),near_zero_fraction=float(np.mean(q<=.01)),
                 top1pct_flow_volume_fraction=safe_ratio(q[q>=qs[-1]].sum(),q.sum()),
                 q_longest_zero_run_days=longest(valid&(y==0)))
        o.update(dict(zip(['q05_mm_day','q10_mm_day','q90_mm_day','q95_mm_day','q99_mm_day'],map(float,qs))))
        pairs=valid[1:]&valid[:-1]&(np.diff(dt).astype(int)==1)
        # Pair-based flashiness: gaps never treated as adjacent days.
        o['flashiness']=safe_ratio(np.abs(np.diff(y))[pairs].sum(),((y[1:]+y[:-1])/2)[pairs].sum())
        months=(dt.astype('datetime64[M]').astype(int)%12)+1
        monthly=[np.mean(y[valid&(months==m)]) for m in range(1,13) if np.any(valid&(months==m))]
        o['q_seasonal_amplitude']=safe_ratio(max(monthly)-min(monthly),mean) if len(monthly)==12 else np.nan
    for j,k in enumerate(['prcp','tmax','tmin','srad','vp','dayl','pet']):
        z=x[:,j]; z=z[np.isfinite(z)]
        o[k+'_mean']=float(z.mean()) if len(z) else np.nan
    p=x[:,0]; pet=x[:,6]; t=(x[:,1]+x[:,2])/2
    good=np.isfinite(p)&(p>=0); pg=p[good]
    o['prcp_cv']=safe_ratio(pg.std(),pg.mean()) if len(pg) else np.nan
    o['prcp_p95_mm_day']=float(np.quantile(pg,.95)) if len(pg) else np.nan
    o['wet_day_fraction']=float(np.mean(pg>=1)) if len(pg) else np.nan
    o['longest_dry_spell_days']=longest(good&(p<1))
    cold=good&np.isfinite(t)
    o['snow_precip_fraction_proxy']=safe_ratio(p[cold&(t<0)].sum(),p[cold].sum())
    o['temperature_mean_degC']=float(np.nanmean(t)) if np.isfinite(t).any() else np.nan
    paired=valid&good
    o['runoff_ratio']=safe_ratio(y[paired].sum(),p[paired].sum())
    o['runoff_ratio_paired_days']=int(paired.sum())
    both=good&np.isfinite(pet)&(pet>=0)
    o['aridity_pet_over_prcp']=safe_ratio(pet[both].sum(),p[both].sum())
    return o

models=[
 ('Daymet_MetOnly','Daymet_MetOnly_Test2005_2009_','Model A Meteorology Only (Daymet)','daymet','1980-01-01','2000-01-01','2005-01-01','2010-01-01'),
 ('Daymet_MetAttr','Daymet_MetAttr_Test2005_2009_','Model A Lumped (Daymet)','daymet','1980-01-01','2000-01-01','2005-01-01','2010-01-01'),
 ('Daymet_PCA10','Daymet_PCA10_Test2005_2009_','Model A Lumped (Daymet + PCA10)','daymet','1980-01-01','2000-01-01','2005-01-01','2010-01-01'),
 ('ERA5_MetAttr','ERA5_MetAttr_Test2005_2009_','Model A Lumped (ERA5)','era5','1980-01-01','2000-01-01','2005-01-01','2010-01-01'),
 ('ERA5_Extended','ERA5_MetAttr_Test2015_2023_','Model A Lumped (ERA5, 1950-2023)','extended','1950-01-01','2010-01-01','2015-01-01','2023-01-04')]
prepared={'daymet':R/'Model A Lumped (Daymet)/prepared','era5':R/'Model A Lumped (ERA5)/prepared',
          'extended':R/'Model A Lumped (ERA5, 1950-2023)/prepared'}
def prediction_path(folder,group,bid,model):
    if model in ('Daymet_MetOnly','Daymet_PCA10'):
        return R/folder/'results_seed42'/('donor_temporal' if group=='Seen' else 'unseen100')/(bid+'_predictions.csv')
    p=R/folder/'run_seed42'
    if group=='Seen':p=p/('temporal_test_2015_latest' if model=='ERA5_Extended' else 'temporal_test_2005_2009')
    return p/(bid+'_predictions.csv')
def error_components(path,expected_n,expected_nse):
    o=dict(prediction_file_present=int(path.exists()))
    if not path.exists(): return o
    records=read(path)
    y=np.array([num(r['observed_mm_day']) for r in records]); q=np.array([num(r['predicted_mm_day']) for r in records])
    good=np.isfinite(y)&np.isfinite(q)&(y>=0); y=y[good];q=q[good]
    assert len(y)==expected_n,(path,len(y),expected_n)
    o['paired_days']=len(y)
    if not len(y):return o
    err=q-y; mse=float(np.mean(err**2)); mean=y.mean(); std=y.std()
    o.update(paired_obs_mean_mm_day=float(mean),paired_obs_std_mm_day=float(std),
             pred_mean_mm_day=float(q.mean()),pred_negative_fraction=float(np.mean(q<0)),
             diagnostic_RMSE=float(np.sqrt(mse)),diagnostic_MAE=float(np.mean(np.abs(err))),
             KGE_beta=safe_ratio(q.mean(),mean),KGE_alpha=safe_ratio(q.std(),std),
             KGE_r=float(np.corrcoef(y,q)[0,1]) if len(y)>1 and std>0 and q.std()>0 else np.nan,
             squared_error_total=float(np.sum(err**2)),NSE_denominator=float(np.sum((y-mean)**2)))
    if std>0:
        calc=1-mse/std**2
        if np.isfinite(expected_nse): assert np.isclose(calc,expected_nse,rtol=2e-4,atol=2e-4),(path,calc,expected_nse)
    lo,hi=np.quantile(y,[.1,.99]); total=np.sum(err**2)
    o['low_flow_squared_error_fraction']=safe_ratio(np.sum(err[y<=lo]**2),total)
    o['high_flow_squared_error_fraction']=safe_ratio(np.sum(err[y>=hi]**2),total)
    return o

output=[]
for i,base in enumerate(comparison):
    bid=base['basin_id'];group=base['seen_unseen']; cache={}
    for source in prepared:
        with np.load(prepared[source]/(bid+'.npz'),allow_pickle=False) as z:
            dates=z['dates'];x=z['x'];y=z['y']
            if source=='extended':ranges=('1950-01-01','2010-01-01','2015-01-01','2023-01-04')
            else:ranges=('1980-01-01','2000-01-01','2005-01-01','2010-01-01')
            cache[source]=({**{'train_'+k:v for k,v in summaries(dates,x,y,*ranges[:2]).items()},
                            **{'test_'+k:v for k,v in summaries(dates,x,y,*ranges[2:]).items()}})
    phys={'attr_'+k:num(a.get(bid)) for k,a in attributes.items()}
    area=phys.get('attr_basin_area',np.nan)
    for key in ('open_water_area_total','reservoir_area_total'):
        phys['attr_'+key.replace('_area_total','_fraction')]=safe_ratio(phys.get('attr_'+key,np.nan),area)
    forest=[num(attributes[k].get(bid)) for k in attributes if k.startswith('lc2_') and 'forest_fraction' in k]
    phys['attr_forest_fraction']=float(np.sum(forest)) if forest and np.isfinite(forest).all() else np.nan
    for model,prefix,folder,source,ts,te,vs,ve in models:
        row={k:base[k] for k in ('basin_id','seen_unseen','catchment_split','country','basin_group','climate_class')}
        row.update(experiment=model,forcing_source=source,train_start=ts,train_end_exclusive=te,test_start=vs,test_end_exclusive=ve)
        for old,new in [('NSE','NSE'),('KGE','KGE'),('N_scored_days','n_scored_days')]:row[new]=num(base[prefix+old])
        row['metric_status']=base[prefix+'Status']; row.update(cache[source]); row.update(phys)
        row.update({'pca_'+k:num(v) for k,v in pca[bid].items() if k.startswith('PC') or k in ('n_subbasins','single_subbasin_flag')})
        n=int(row['n_scored_days']) if np.isfinite(row['n_scored_days']) else 0
        row.update(error_components(prediction_path(folder,group,bid,model),n,row['NSE']))
        row['test_q_mean_m3_s']=safe_ratio(row['test_q_mean_mm_day']*area,86.4)
        row['test_train_q_mean_ratio']=safe_ratio(row['test_q_mean_mm_day'],row['train_q_mean_mm_day'])
        row['test_train_prcp_mean_ratio']=safe_ratio(row['test_prcp_mean'],row['train_prcp_mean'])
        output.append(row)
    if (i+1)%100==0:print('EXTRACTED',i+1,'/1426',flush=True)
headers=list(dict.fromkeys(k for r in output for k in r))
tmp=TARGET.with_suffix('.tmp.csv')
with tmp.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(output)
assert len(output)==7130 and len({(r['basin_id'],r['experiment']) for r in output})==7130
tmp.replace(TARGET)
info=dict(rows=7130,basins=1426,experiments=5,columns=len(headers),attribute_units=units,
          missing_selected_attributes=sorted(set(selected)-set(attributes)),
          raw_attribute_source=str(attr_path),comparison_source=str(R/'LSTM_Performance_Comparison/basin_lstm_performance_comparison.csv'),
          csv_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest())
(B/'preparation_verification.json').write_text(json.dumps(info,indent=2))
print('PREPARATION COMPLETE',json.dumps(info),flush=True)
