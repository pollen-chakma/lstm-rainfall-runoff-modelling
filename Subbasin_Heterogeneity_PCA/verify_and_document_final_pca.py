from pathlib import Path
import csv,json,collections
import numpy as np
B=Path("/home/pollen26/Desktop/LSTM_RR_Modelling/Subbasin_Heterogeneity_PCA");A=B/'final_pca'
s=json.loads((A/'pca_summary.json').read_text());spec=list(csv.DictReader((A/'final_preprocessing.csv').open()));idx=list(csv.DictReader((B/'subbasin_index.csv').open()));audit=list(csv.DictReader((B/'basin_subbasin_audit.csv').open()))
raw=np.load(B/'subbasin_attribute_matrix.npz');X=raw['values'];names=raw['attributes'].astype(str).tolist();area=np.array([float(r['area_km2']) for r in idx]);model=np.load(A/'pca_transform.npz');V=[]
for r in spec:
 v=X[:,names.index(r['attribute'])].copy();n=r['normalization'];name=r['attribute']
 if n=='divide_by_subbasin_area_km2':v/=area
 elif n in ('1000_times_million_m3_divided_by_km2','1000_times_count_divided_by_km2'):v=1000*v/area
 bad=~np.isfinite(v)
 if r['category'] in ('Soil','Open water') and 'log_' not in name:bad|=v<0
 if name.endswith('_fraction'):bad|=(v < -1e-8)|(v>1+1e-8)
 v[bad]=float(r['imputation_value']);t=r['transform'];ref=float(r['transform_reference'])
 if t=='sqrt_fraction':v=np.sqrt(np.clip(v,0,1))
 elif t.startswith('log1p'):v=np.log1p(v/ref)
 elif t.startswith('asinh'):v=np.arcsinh(v/ref)
 V.append((v-float(r['mean']))/float(r['std'])*float(r['domain_factor']))
q=np.column_stack(V)@model['components'];saved=np.load(A/'subbasin_pc_scores.npz')['scores'];assert np.allclose(q,saved,atol=1e-12)
checks={}
for K in (10,20):
 rows=list(csv.DictReader((B/f'basin_pca{K}_mean_std_features.csv').open()));assert len(rows)==1426
 for b,r in zip(audit,rows):
  lo,hi=int(b['start_row']),int(b['end_row_exclusive']);w=area[lo:hi]/area[lo:hi].sum();p=q[lo:hi,:K];m=np.average(p,axis=0,weights=w);sd=np.sqrt(np.average((p-m)**2,axis=0,weights=w))
  assert r['basin_id']==b['basin_id']
  assert np.allclose(m,[float(r[f'PC{j+1:02d}_mean']) for j in range(K)],atol=1e-10)
  assert np.allclose(sd,[float(r[f'PC{j+1:02d}_std']) for j in range(K)],atol=1e-10)
 checks[str(K)]={'rows':len(rows),'columns':len(rows[0]),'split':dict(collections.Counter(r['seen_unseen'] for r in rows))}
(A/'independent_reapplication_verification.json').write_text(json.dumps({'passed':True,'saved_transform_recreates_all_subbasin_scores':True,'independent_aggregation_matches_both_csvs':True,'files':checks},indent=2))
text=f'''# Final subbasin PCA inputs

Completed on Nibi. One shared PCA was fitted only to the 30,604 subbasin records of the 1,326 donor basins, then applied unchanged to all 32,802 records belonging to 1,426 basins. The 100 held-out basins contribute no fitting weight. No discharge data were used.

## Files for model input
- basin_pca10_mean_std_features.csv: 1,426 rows; 20 PC features (10 means + 10 standard deviations), plus five identifying/flag columns.
- basin_pca20_mean_std_features.csv: 1,426 rows; 40 PC features (20 means + 20 standard deviations), plus the same five columns.
- single_subbasin_flag is an additional recommended model input; 304 basins have one subbasin and therefore zero between-subbasin standard deviations. This does not imply no intrinsic heterogeneity.
- basin_id is the join key. seen_unseen, catchment_split and n_subbasins are metadata, not default model predictors. Read IDs as strings.

## Method
The screening starts from 1,101 subbasin attributes (not the 1,178 whole-basin attributes). The physical-property shortlist contained 317 candidates. Final PCA uses {s['final_attributes']} attributes: {s['retained_domains']}.
Climate, hydrological response, identifiers, raw circular aspect, quality/uncertainty fields and network upstream area are excluded. Counts and extensive lake/reservoir/stream measures are converted to area-normalized measures. Fraction fields use a square-root transformation; sufficiently skewed fields use a training-derived log1p or signed asinh transformation. Invalid values become missing and are imputed with donor basin-area-weighted medians. Full per-attribute details are in final_pca/final_preprocessing.csv and final_attribute_decisions.csv.
Redundancy removal uses donor-only weighted rank correlation >=0.98 only within the same min/mean/max property, source and soil depth; useful standard deviations and separate depths are preserved.
Each donor basin has equal total fitting weight, distributed among its subbasins by area. Features are standardized using these weights, then divided by the square root of the number of retained features in their physical domain. This gives each domain equal initial total variance. Consequently explained variance refers to this transformed, domain-balanced space.
The 10-PC set explains {100*s['pca10_cumulative_explained_variance']:.2f}% and the 20-PC set {100*s['pca20_cumulative_explained_variance']:.2f}%. The first 10 PCs are identical between files.
For each basin, PC_mean=sum(area_fraction*PC_score), and PC_std=sqrt(sum(area_fraction*(PC_score-PC_mean)^2)); this is a population standard deviation, not an uncertainty estimate. These summaries capture average properties and dispersion, but do not encode spatial arrangement or routing.

## Reuse and reproducibility
Join one of these files to daily forcing/discharge by basin_id. Use the PC columns and optionally single_subbasin_flag as static LSTM inputs. Fit any subsequent LSTM static-feature scaler using donor basins only. Keep the existing donor/unseen IDs and chronological splits. Do not refit PCA on test basins.
final_pca/model_input_columns_pca10.json and pca20.json list exact model and metadata columns. final_pca/pca_transform.npz stores the fixed numerical transform; pca_loadings.csv gives interpretation and explained_variance.csv gives every component. subbasin_pc_scores.npz follows subbasin_index.csv row order.
fit_final_pca.py is the complete fitting/export source. It refuses to overwrite existing final CSVs. For a fresh recomputation, copy the source matrix, index, basin audit, reduction plan, inventory and script into a separate directory, then run the script with NumPy installed. The notebook 02_Final_Subbasin_PCA.ipynb displays and explains the completed outputs without refitting by default.
Verification includes finite values, all basin IDs, 1,326/100 split preservation, zero unseen fitting weight, orthogonality and weighted score covariance, area-weighted variance identities, shared first 10 PCs, and independent reapplication of the saved transformation and aggregation to all rows.

## Interpretation limits
Physical source maps represent their own reference dates; this is a retrospective static-property experiment. Existing audit reports 440 raw subbasin IDs shared across donor and held-out basins; these can represent nested or overlapping geography, so the held-out set is unseen by basin ID but not guaranteed spatially disjoint. The original split has been preserved. Composite basin_id/subbasin_id identifies records. The 12 anomalous RDRS temperature standard-deviation fields found in exploration are outside this physical PCA.
No new LSTM has been trained in this step. These files are static inputs for the next model experiment.
'''
(B/'FINAL_PCA_README.md').write_text(text)
nb=json.loads("{\"cells\":[{\"cell_type\":\"markdown\",\"metadata\":{},\"source\":[\"# Final subbasin PCA\\n\",\"See FINAL_PCA_README.md for the full method and reuse instructions. This notebook inspects the completed outputs without refitting.\"]},{\"cell_type\":\"code\",\"metadata\":{},\"execution_count\":null,\"outputs\":[],\"source\":\"from pathlib import Path\\nimport json, pandas as pd\\nB=Path.cwd()\\nsummary=json.loads((B/\\\"final_pca/pca_summary.json\\\").read_text())\\nprint({k:v for k,v in summary.items() if not k.endswith(\\\"_ids\\\") and \\\"sha256\\\" not in k})\"},{\"cell_type\":\"code\",\"metadata\":{},\"execution_count\":null,\"outputs\":[],\"source\":\"features10=pd.read_csv(B/\\\"basin_pca10_mean_std_features.csv\\\",dtype={\\\"basin_id\\\":str})\\nfeatures20=pd.read_csv(B/\\\"basin_pca20_mean_std_features.csv\\\",dtype={\\\"basin_id\\\":str})\\ndisplay(features10.head())\\ndisplay(features20.head())\\nprint(features20.groupby(\\\"seen_unseen\\\").size())\"},{\"cell_type\":\"code\",\"metadata\":{},\"execution_count\":null,\"outputs\":[],\"source\":\"display(pd.read_csv(B/\\\"final_pca/final_preprocessing.csv\\\"))\\ndisplay(pd.read_csv(B/\\\"final_pca/explained_variance.csv\\\").head(20))\\nprint((B/\\\"final_pca/independent_reapplication_verification.json\\\").read_text())\"}],\"metadata\":{\"kernelspec\":{\"display_name\":\"Python 3\",\"language\":\"python\",\"name\":\"python3\"},\"language_info\":{\"name\":\"python\"}},\"nbformat\":4,\"nbformat_minor\":5}")
(B/'02_Final_Subbasin_PCA.ipynb').write_text(json.dumps(nb,indent=1))
for c in nb['cells']:
 if c['cell_type']=='code':compile(c['source'],'notebook','exec')
print('INDEPENDENT VERIFICATION PASSED',checks)
