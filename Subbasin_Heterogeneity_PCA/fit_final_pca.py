from pathlib import Path
import csv,json,time,hashlib,collections,math,os
import numpy as np
B=Path(__file__).resolve().parent;A=B/'final_pca';A.mkdir(exist_ok=True)
OUTPUTS=[B/'basin_pca10_mean_std_features.csv',B/'basin_pca20_mean_std_features.csv']
assert not any(p.exists() for p in OUTPUTS),'Final CSV exists; preserve it before intentionally recomputing.'
def read(name):
 with open(B/name,newline='') as f:return list(csv.DictReader(f))
def writecsv(path,rows,fields=None):
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def js(path,obj):path.write_text(json.dumps(obj,indent=2))
def wm(x,w):
 good=np.isfinite(x)&(w>0);x=x[good];w=w[good];assert len(x)
 order=np.argsort(x,kind='stable');x=x[order];w=w[order];return float(x[np.searchsorted(np.cumsum(w),w.sum()/2)])
def moments(x,w):
 mu=float(w@x);sd=float(np.sqrt(w@((x-mu)**2)));sk=float(w@(((x-mu)/sd)**3)) if sd>0 else 0.;return mu,sd,sk
start=time.time();idx=read('subbasin_index.csv');basins=read('basin_subbasin_audit.csv');plan=read('feature_reduction_plan.csv');inventory={r['attribute']:r for r in read('attribute_inventory_and_screening.csv')}
with np.load(B/'subbasin_attribute_matrix.npz') as f:X=f['values'];names=f['attributes'].astype(str).tolist()
assert X.shape==(32802,1101) and len(idx)==32802 and len(basins)==1426
col={n:i for i,n in enumerate(names)};tm=np.array([r['catchment_split']=='training_donor' for r in idx]);area=np.array([float(r['area_km2']) for r in idx]);W=np.zeros(len(idx));local=np.zeros(len(idx));donor_ids=[];held_ids=[]
for b in basins:
 lo,hi=int(b['start_row']),int(b['end_row_exclusive']);assert hi-lo==int(b['subbasins']);assert all(r['basin_id']==b['basin_id'] for r in idx[lo:hi]);local[lo:hi]=area[lo:hi]/area[lo:hi].sum()
 if b['catchment_split']=='training_donor':W[lo:hi]=local[lo:hi]/1326;donor_ids.append(b['basin_id'])
 else:held_ids.append(b['basin_id'])
assert len(donor_ids)==1326 and len(held_ids)==100 and tm.sum()==30604 and (~tm).sum()==2198 and np.isclose(W.sum(),1.) and not W[~tm].any()
w=W[tm];assert np.isclose(w.sum(),1.)
selected=[r for r in plan if r['initial_physical_pca_candidate']=='True'];assert len(selected)==317
anomaly_names={r['attribute'] for r in read('data_quality_anomalies.csv')}
assert not anomaly_names&{r['attribute'] for r in selected}
features=[];specs=[];decisions=[]
for row in selected:
 name=row['attribute'];v=X[:,col[name]].copy();cat=row['category'];unit=row['unit'];normalization='identity';feature=name
 if name=='upstream_area':decisions.append(dict(attribute=name,action='exclude_network_context',reason='Upstream contributing area is not an intrinsic local subbasin property.'));continue
 if name=='stream_length':v=v/area;feature='stream_length_density';unit='km per km2';normalization='divide_by_subbasin_area_km2'
 elif cat=='Open water':
  if '_area_' in name:v=v/area;feature=name+'_per_basin_area';unit='km2 per km2';normalization='divide_by_subbasin_area_km2'
  elif '_volume_' in name:v=1000*v/area;feature=name+'_equivalent_mm';unit='mm';normalization='1000_times_million_m3_divided_by_km2'
  elif name in ('open_water_number','known_reservoirs'):v=1000*v/area;feature=name+'_per_1000_km2';unit='count per 1000 km2';normalization='1000_times_count_divided_by_km2'
 invalid=~np.isfinite(v)
 if cat in ('Soil','Open water') and 'log_' not in name:invalid |= v<0
 is_fraction=name.endswith('_fraction')
 if is_fraction:invalid |= (v < -1e-8)|(v>1+1e-8)
 v[invalid]=np.nan
 med=wm(v[tm],w);imputed=np.where(np.isfinite(v),v,med);t=imputed[tm];mu,sd,sk=moments(t,w)
 transform='identity';reference=1.
 if is_fraction:imputed=np.sqrt(np.clip(imputed,0,1));transform='sqrt_fraction'
 elif 'log_permeability' not in name and (abs(sk)>2 or 'conductivity_' in name):
  nz=np.abs(t)>0
  if nz.any():reference=wm(np.abs(t[nz]),w[nz]);reference=max(reference,np.finfo(float).tiny)
  if t.min()>=0 and not name.startswith('elev_'):imputed=np.log1p(imputed/reference);transform='log1p_over_training_positive_median'
  else:imputed=np.arcsinh(imputed/reference);transform='asinh_over_training_absolute_median'
 transformed_train=imputed[tm];mu,sd,_=moments(transformed_train,w)
 if sd<=1e-12*max(1.,float(np.max(np.abs(transformed_train)))):
  decisions.append(dict(attribute=name,action='exclude_numerically_constant_after_transform',reason='No stable numerical variation for standardization.'));continue
 specs.append(dict(attribute=name,feature=feature,category=cat,source=row['source'],source_unit=row['unit'],processed_unit=unit,normalization=normalization,imputation='training_basin_area_weighted_median',imputation_value=med,transform=transform,transform_reference=reference,mean=mu,std=sd,missing_training_rows=int(invalid[tm].sum()),missing_unseen_rows=int(invalid[~tm].sum())))
 features.append(imputed)
V=np.column_stack(features);del features;print('TRANSFORMED',V.shape,flush=True)
# Weighted midrank correlations, fitted only on donor subbasins.
T=V[tm];rank=np.empty_like(T)
for j in range(T.shape[1]):
 unique,inv=np.unique(T[:,j],return_inverse=True);mass=np.bincount(inv,weights=w);mid=np.cumsum(mass)-.5*mass;r=mid[inv];r-=w@r;r/=np.sqrt(w@(r*r));rank[:,j]=r
corr=rank.T@(w[:,None]*rank)
keep=[];families=collections.defaultdict(list);redundant=[]
order=sorted(range(len(specs)),key=lambda j:(not specs[j]['attribute'].endswith('_mean'),specs[j]['attribute']))
for j in order:
 name=specs[j]['attribute'];parts=name.rsplit('_',1);stat=parts[-1];family=(specs[j]['category'],specs[j]['source'],parts[0]) if stat in ('min','mean','max') else None
 rep=None
 if family:
  rep=next((k for k in families[family] if abs(corr[j,k])>=.98),None)
 # A length/area alias is redundant with the supplied drainage density only if verified numerically.
 if name=='stream_length':
  aliases=[k for k in keep if specs[k]['attribute']=='streamdensity' and np.allclose(T[:,j],T[:,k],rtol=1e-10,atol=1e-12)]
  if aliases:rep=aliases[0]
 if rep is not None:
  redundant.append(dict(attribute=name,retained_attribute=specs[rep]['attribute'],weighted_spearman=float(np.clip(corr[j,rep],-1,1)),reason='Redundant min/mean/max within the same property, depth and source, or verified drainage-density alias.'))
  decisions.append(dict(attribute=name,action='exclude_redundant_same_property',reason='Retain '+specs[rep]['attribute']+'; weighted absolute rank correlation >= 0.98.'))
 else:
  keep.append(j)
  if family:families[family].append(j)
keep=sorted(keep);V=V[:,keep];specs=[specs[j] for j in keep];assert len(specs)>=20
# Unit-variance scaling, then equal total initial variance for each physical domain.
mu=np.array([s['mean'] for s in specs]);sd=np.array([s['std'] for s in specs]);counts=collections.Counter(s['category'] for s in specs);factor=np.array([1/np.sqrt(counts[s['category']]) for s in specs]);Z=(V-mu)/sd*factor
assert np.isfinite(Z).all();center=w@Z[tm];C=Z[tm].T@(w[:,None]*Z[tm]);C=(C+C.T)/2
assert np.max(np.abs(center))<1e-9
val,vec=np.linalg.eigh(C);order=np.argsort(val)[::-1];val=np.maximum(val[order],0);vec=vec[:,order]
for j in range(vec.shape[1]):
 k=int(np.argmax(np.abs(vec[:,j])))
 if vec[k,j]<0:vec[:,j]*=-1
components=vec[:,:20];ratio=val/val.sum();scores=Z@components
assert np.allclose(components.T@components,np.eye(20),atol=1e-10)
assert np.allclose(scores[tm].T@(w[:,None]*scores[tm]),np.diag(val[:20]),atol=1e-8)
# Held-out properties never enter any fitted parameter; the same fixed transformation is applied to them.
means=[];stds=[];basin_metadata=[]
for b in basins:
 lo,hi=int(b['start_row']),int(b['end_row_exclusive']);q=scores[lo:hi];a=local[lo:hi];mean=a@q;variance=a@((q-mean)**2);st=np.sqrt(np.maximum(variance,0))
 if hi-lo==1:st[:]=0
 means.append(mean);stds.append(st);basin_metadata.append(dict(basin_id=b['basin_id'],seen_unseen='Seen' if b['catchment_split']=='training_donor' else 'Unseen',catchment_split=b['catchment_split'],n_subbasins=hi-lo,single_subbasin_flag=int(hi-lo==1)))
means=np.array(means);stds=np.array(stds);assert np.isfinite(means).all() and np.isfinite(stds).all() and (stds>=0).all()
# Independent moment identity verifies population area-weighted dispersions.
for i,b in enumerate(basins):
 lo,hi=int(b['start_row']),int(b['end_row_exclusive']);raw=local[lo:hi]@(scores[lo:hi]**2)-means[i]**2
 assert np.allclose(stds[i]**2,raw,atol=1e-9)
assert sum(m['single_subbasin_flag'] for m in basin_metadata)==304
# Hand-calculated heterogeneity checks.
assert np.allclose(np.array([.5,.5])@np.array([-2.,2.]),0.) and np.isclose(np.sqrt(np.array([.5,.5])@(np.array([-2.,2.])**2)),2.)
for K,path in zip((10,20),OUTPUTS):
 cols=[f'PC{j+1:02d}_{stat}' for j in range(K) for stat in ('mean','std')];rows=[]
 for i,m in enumerate(basin_metadata):
  row=dict(m)
  for j in range(K):row[f'PC{j+1:02d}_mean']=float(means[i,j]);row[f'PC{j+1:02d}_std']=float(stds[i,j])
  rows.append(row)
 tmp=path.with_suffix('.tmp.csv');writecsv(tmp,rows)
 check=read(tmp.name);assert len(check)==1426 and len(check[0])==5+2*K
 assert collections.Counter(r['seen_unseen'] for r in check)=={'Seen':1326,'Unseen':100}
 assert all(np.isfinite(float(r[c])) for r in check for c in cols)
 assert [r['basin_id'] for r in check]==[b['basin_id'] for b in basins]
 js(A/f'model_input_columns_pca{K}.json',dict(pca_feature_columns=cols,auxiliary_model_feature='single_subbasin_flag',recommended_static_model_inputs=cols+['single_subbasin_flag'],metadata_only_columns=['basin_id','seen_unseen','catchment_split','n_subbasins'],join_key='basin_id',pca_components=K))
# Both dimensionalities must use identical first-ten scores.
a=read(OUTPUTS[0].with_suffix('.tmp.csv').name);b=read(OUTPUTS[1].with_suffix('.tmp.csv').name)
assert all(all(x[k]==y[k] for k in x) for x,y in zip(a,b))
for p in OUTPUTS:p.with_suffix('.tmp.csv').replace(p)
for s,f in zip(specs,factor):s['domain_factor']=float(f);decisions.append(dict(attribute=s['attribute'],action='retained_final',reason='Transformed, training-imputed/scaled and domain-balanced.'))
writecsv(A/'final_preprocessing.csv',specs);writecsv(A/'final_attribute_decisions.csv',decisions);writecsv(A/'redundancy_removals.csv',redundant,['attribute','retained_attribute','weighted_spearman','reason'])
writecsv(A/'explained_variance.csv',[dict(component=i+1,eigenvalue=float(v),explained_variance_ratio=float(ratio[i]),cumulative_explained_variance=float(ratio[:i+1].sum())) for i,v in enumerate(val)])
writecsv(A/'pca_loadings.csv',[dict(feature=s['feature'],original_attribute=s['attribute'],category=s['category'],**{f'PC{j+1:02d}':float(components[i,j]) for j in range(20)}) for i,s in enumerate(specs)])
np.savez_compressed(A/'pca_transform.npz',attributes=np.array([s['attribute'] for s in specs]),features=np.array([s['feature'] for s in specs]),imputation_values=np.array([s['imputation_value'] for s in specs]),transform_names=np.array([s['transform'] for s in specs]),transform_reference=np.array([s['transform_reference'] for s in specs]),normalization_names=np.array([s['normalization'] for s in specs]),mean=mu,std=sd,domain_factor=factor,components=components,eigenvalues=val,explained_variance_ratio=ratio)
np.savez_compressed(A/'subbasin_pc_scores.npz',scores=scores,basin_ids=np.array([r['basin_id'] for r in idx]),subbasin_ids=np.array([r['subbasin_id'] for r in idx]))
inputs=['subbasin_attribute_matrix.npz','subbasin_index.csv','basin_subbasin_audit.csv','feature_reduction_plan.csv']
hashes={}
for name in inputs:
 h=hashlib.sha256()
 with open(B/name,'rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 hashes[name]=h.hexdigest()
summary=dict(status='complete',method='shared physical-domain-balanced, basin-balanced area-weighted PCA',initial_candidates=317,final_attributes=len(specs),retained_domains=dict(counts),fit_basins=1326,held_out_basins=100,training_subbasin_records=int(tm.sum()),unseen_subbasin_records=int((~tm).sum()),output_basins=1426,components=20,pca10_cumulative_explained_variance=float(ratio[:10].sum()),pca20_cumulative_explained_variance=float(ratio[:20].sum()),single_subbasin_basins=304,correlation_threshold=.98,correlation_policy='Within min/mean/max family at fixed property/depth/source; standard deviations and different depths preserved.',input_sha256=hashes,output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUTS},fit_basin_ids=donor_ids,unseen_basin_ids=held_ids,seconds=round(time.time()-start),numpy_version=np.__version__)
js(A/'pca_summary.json',summary);js(A/'verification.json',dict(passed=True,all_1426_basins_retained=True,split_1326_100_preserved=True,all_predictors_finite=True,unseen_fit_weight_zero=True,one_shared_pca=True,orthonormal_components=True,weighted_score_covariance_verified=True,area_weighted_variance_identity_verified=True,first_ten_components_identical_between_files=True,single_subbasin_stds_zero=True,pca_fit_uses_no_discharge=True))
print('PCA COMPLETE',json.dumps({k:v for k,v in summary.items() if not k.endswith('_ids') and 'sha256' not in k},indent=2),flush=True)
