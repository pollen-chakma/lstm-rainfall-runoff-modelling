from pathlib import Path
import csv,json,time,hashlib,collections,os
import numpy as np
R=Path('/project/6102174/data/misc-data/CAMELS-SPAT');B=Path.home()/'Desktop/LSTM_RR_Modelling/Subbasin_Heterogeneity_PCA';B.mkdir(exist_ok=True)
def savecsv(name,rows,fields=None):
 with open(B/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def savejson(name,obj):(B/name).write_text(json.dumps(obj,indent=2))
with open(B.parent/'Model A Lumped (Daymet)/split_manifest.csv') as f:manifest=list(csv.DictReader(f))
assert len(manifest)==1426 and collections.Counter(r['split'] for r in manifest)=={'training_donor':1326,'unseen_test':100}
with open(R/'attributes/attributes-lumped.csv') as f:
 reader=csv.reader(f);lhead=next(reader);lmeta=[];larea={}
 for row in reader:
  lmeta.append(row[:4])
  if row[1]=='basin_area':larea={b:float(v) for b,v in zip(lhead[4:],row[4:])}
paths={p.name.removesuffix('_attributes.csv'):p for p in (R/'attributes').glob('*/*_attributes.csv')}
assert set(paths)=={r['basin_id'] for r in manifest}
nonnumeric=collections.defaultdict(collections.Counter)
matrices=[];indices=[];basins=[];schema=None;bad=[];masked=[];start=time.time();cursor=0
for k,m in enumerate(manifest):
 bid=m['basin_id'];p=paths[bid]
 with open(p) as f:reader=csv.reader(f);h=next(reader);rows=list(reader)
 meta=[r[:4] for r in rows];ids=h[4:]
 if schema is None:schema=meta
 assert meta==schema,(bid,'attribute schema/order changed')
 assert len(ids)==len(set(ids)) and len(ids)>0,(bid,'duplicate/empty subbasin IDs')
 values=np.full((len(ids),len(meta)),np.nan,dtype='float64')
 for j,row in enumerate(rows):
  assert len(row)==len(h),(bid,j,'ragged row')
  for i,v in enumerate(row[4:]):
   try:values[i,j]=float(v)
   except ValueError:
    if v.strip()=='--':masked.append(dict(basin_id=bid,subbasin_id=ids[i],catchment_split=m['split'],attribute=row[1],value=v))
    if v.strip().lower() not in ('','nan','na','n/a','none','null','--'):
     bad.append(dict(basin_id=bid,subbasin_id=ids[i],catchment_split=m['split'],attribute=row[1],value=v))
     if m['split']=='training_donor':nonnumeric[row[1]][v]+=1
 values[~np.isfinite(values)]=np.nan
 jarea=next(j for j,s in enumerate(meta) if s[1]=='basin_area');area=values[:,jarea].astype('float64')
 assert np.isfinite(area).all() and (area>0).all(),(bid,'invalid subbasin areas')
 weights=area/area.sum();train=m['split']=='training_donor';n=len(ids)
 for i,sub in enumerate(ids):indices.append(dict(basin_id=bid,subbasin_id=sub,catchment_split=m['split'],seen_unseen='Seen' if train else 'Unseen',area_km2=float(area[i]),within_basin_area_weight=float(weights[i]),pca_fit_weight=float(weights[i]/1326) if train else 0.))
 basins.append(dict(basin_id=bid,catchment_split=m['split'],basin_group=m['basin_group'],subbasins=n,start_row=cursor,end_row_exclusive=cursor+n,area_sum_km2=float(area.sum()),lumped_area_km2=larea[bid],area_relative_difference=float((area.sum()-larea[bid])/larea[bid]),effective_subbasins=float(1/np.sum(weights**2)),single_subbasin=int(n==1)))
 cursor+=n;matrices.append(values)
 if (k+1)%100==0:print('SCANNED',k+1,'BASINS',cursor,'SUBBASINS',flush=True)
X=np.concatenate(matrices);del matrices
savecsv('masked_numeric_values.csv',masked,['basin_id','subbasin_id','catchment_split','attribute','value'])
savecsv('categorical_attribute_values.csv',bad,['basin_id','subbasin_id','catchment_split','attribute','value'])
savecsv('subbasin_index.csv',indices);savecsv('basin_subbasin_audit.csv',basins)
np.savez_compressed(B/'subbasin_attribute_matrix.npz',values=X,attributes=np.array([s[1] for s in schema]))
trainmask=np.array([r['seen_unseen']=='Seen' for r in indices]);T=X[trainmask];W=np.array([r['pca_fit_weight'] for r in indices])[trainmask];assert np.isclose(W.sum(),1.)
fitids=[r['basin_id'] for r in manifest if r['split']=='training_donor'];holdids=[r['basin_id'] for r in manifest if r['split']=='unseen_test']
assert len(fitids)==1326 and len(holdids)==100
lookup={(r['basin_id'],r['subbasin_id']):float(r['pca_fit_weight']) for r in indices}
cat_mass=collections.Counter()
for r in bad:
 if r['catchment_split']=='training_donor':cat_mass[r['attribute']]+=lookup[(r['basin_id'],r['subbasin_id'])]

inventory=[];duplicate_hashes={};duplicate_groups=[]
for j,(cat,name,unit,source) in enumerate(schema):
 v=T[:,j];good=np.isfinite(v);n=int(good.sum());a=v[good];p=W[good];numeric_missing=float(1-p.sum());missing=float(np.clip(numeric_missing-cat_mass[name],0.,1.));mean=float(np.sum(a*p)/p.sum()) if n else np.nan;sd=float(np.sqrt(np.sum(p*(a-mean)**2)/p.sum())) if n else np.nan
 mn=float(a.min()) if n else np.nan;mx=float(a.max()) if n else np.nan;span=mx-mn;exact=bool(n and mn==mx);near=bool(n and span<=1e-12*max(1.,abs(mn),abs(mx)))
 flags=[];role='physical_candidate';reason='Review physical meaning and redundancy; retain variability measures.'
 if cat=='Climate':role='climate_variant_review';reason='Separate climate from primary physical-property PCA; RDRS statistics may span beyond training years.'
 if name.startswith('num_years'):role='exclude_record_length';reason='Record availability, not a physical catchment property.'
 if name=='basin_area':role='area_weight_only';reason='Use for area weights; avoid encoding subbasin discretization in primary PCA.'
 if name in ('centroid_lat','centroid_lon'):role='location_context_only';reason='Geographic position is separate from physical-property heterogeneity.'
 if 'aspect' in name:role='circular_variable_review';reason='Angles need circular treatment; do not enter raw degree values into linear PCA.'
 if 'uncertainty' in name.lower() or 'no_data' in name or 'unclassified' in name:role='data_quality_review';reason='Data coverage or estimation uncertainty; retain as diagnostics before deciding on predictors.'
 if 'fraction' in name:flags.append('compositional_fraction_review')
 if name in ('open_water_number','known_reservoirs') or name.endswith('_total'):flags.append('extensive_variable_area_normalization_review')
 if cat=='Soil':flags.append('preserve_depth_and_spatial_statistic_meanings')
 if name.endswith('_std'):flags.append('potential_within_subbasin_heterogeneity_descriptor')
 if 'log_permeability' in name:flags.append('already_log_transformed_do_not_log_again')
 action='review'
 if nonnumeric[name]:action='review_categorical_encoding';role='categorical_season_review';reason='Non-numeric categories must be encoded appropriately before PCA; not numeric missing values.'
 elif not n:action='exclude_all_missing'
 elif exact:action='exclude_constant_training'
 elif near:action='review_numerically_near_constant'
 elif missing>.4:action='review_high_missingness'
 duplicate_of=''
 if n and not exact:
  vv=v.copy();vv[np.isnan(vv)]=np.nan;vv[vv==0]=0;digest=(unit,hashlib.sha256(vv.tobytes()).hexdigest())
  if digest in duplicate_hashes:duplicate_of=duplicate_hashes[digest];duplicate_groups.append(dict(attribute=name,duplicate_of=duplicate_of,unit=unit));action='review_exact_duplicate'
  else:duplicate_hashes[digest]=name
 inventory.append(dict(attribute=name,category=cat,unit=unit,source=source,training_subbasins=len(T),valid_training_subbasins=n+sum(nonnumeric[name].values()),valid_numeric_training_subbasins=n,missing_fraction_numeric_rows=float(1-n/len(T)),missing_fraction_numeric_basin_area_weighted=numeric_missing,missing_fraction_rows=float(np.clip(1-(n+sum(nonnumeric[name].values()))/len(T),0.,1.)),missing_fraction_basin_area_weighted=missing,training_min=mn,training_max=mx,training_weighted_mean=mean,training_weighted_std=sd,constant_training=exact,numerically_near_constant=near,duplicate_of=duplicate_of,domain_role=role,quality_action=action,domain_reason=reason,nonnumeric_training_rows=sum(nonnumeric[name].values()),training_category_counts=json.dumps(dict(nonnumeric[name])),review_flags=';'.join(flags)))
savecsv('attribute_inventory_and_screening.csv',inventory)
savecsv('exact_duplicate_candidates.csv',duplicate_groups,['attribute','duplicate_of','unit'])
names={s[1] for s in schema};only=[dict(category=s[0],attribute=s[1],unit=s[2],source=s[3]) for s in lmeta if s[1] not in names];savecsv('whole_basin_only_attributes.csv',only)
# Diagnostic rank correlations: 8 area-proportional samples per donor; no holdout data used.
rng=np.random.default_rng(42);selected=[]
for b in basins:
 if b['catchment_split']!='training_donor':continue
 lo,hi=b['start_row'],b['end_row_exclusive'];a=X[lo:hi,jarea].astype('float64');cdf=np.cumsum(a/a.sum());u=(np.arange(8)+rng.random(8))/8;selected.extend((lo+np.searchsorted(cdf,u)).tolist())
S=X[np.array(selected)].astype('float64');usable=[]
for j in range(S.shape[1]):
 v=S[:,j];g=np.isfinite(v)
 if g.sum()<2:continue
 v[~g]=np.median(v[g]);unique,inv,counts=np.unique(v,return_inverse=True,return_counts=True)
 if len(unique)<2:continue
 ends=np.cumsum(counts);ranks=(ends-counts+(ends-1))/2.;v[:]=ranks[inv];v-=v.mean();norm=np.sqrt(np.sum(v*v));v/=norm;usable.append(j)
Z=S[:,usable];C=Z.T@Z;pairs=[]
for a,b in zip(*np.where(np.triu(np.abs(C)>=.98,1))):
 i,j=usable[a],usable[b];pairs.append(dict(attribute_1=schema[i][1],attribute_2=schema[j][1],spearman_r=float(C[a,b]),category_1=schema[i][0],category_2=schema[j][0],diagnostic='training_only_balanced_area_sample_not_automatic_removal'))
savecsv('high_correlation_review_pairs.csv',pairs,['attribute_1','attribute_2','spearman_r','category_1','category_2','diagnostic'])
summary=dict(phase='exploration_complete',basins=1426,training_basins=1326,unseen_basins=100,subbasin_records=len(X),training_subbasin_records=int(trainmask.sum()),unseen_subbasin_records=int((~trainmask).sum()),subbasin_attributes=len(schema),whole_basin_attributes=len(lmeta),whole_basin_only_attributes=len(only),subbasin_count_min=min(b['subbasins'] for b in basins),subbasin_count_max=max(b['subbasins'] for b in basins),single_subbasin_basins=sum(b['single_subbasin'] for b in basins),area_sum_relative_error_max=max(abs(b['area_relative_difference']) for b in basins),categories=dict(collections.Counter(s[0] for s in schema)),sources=dict(collections.Counter(s[3] for s in schema)),quality_actions=dict(collections.Counter(r['quality_action'] for r in inventory)),domain_roles=dict(collections.Counter(r['domain_role'] for r in inventory)),exact_duplicate_candidates=len(duplicate_groups),categorical_attributes=sum(bool(c) for c in nonnumeric.values()),categorical_value_records=len(bad),masked_numeric_records=len(masked),masked_numeric_attributes=len({r["attribute"] for r in masked}),correlation_review_pairs=len(pairs),correlation_absolute_threshold=.98,correlation_training_sample_size=len(selected),pca_fit=False,seconds=round(time.time()-start))
savejson('exploration_summary.json',summary);savejson('fit_scope.json',dict(training_basin_ids=fitids,unseen_basin_ids=holdids,fitted_pca=False,quality_statistics_scope='training_basins_only',fit_weight='within_basin_area_fraction / 1326; equal total mass per training basin',preserve_source_data=True))
print(json.dumps(summary,indent=2),flush=True)
