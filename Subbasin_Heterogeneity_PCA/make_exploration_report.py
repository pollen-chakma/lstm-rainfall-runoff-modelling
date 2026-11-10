from pathlib import Path
import json,csv,collections,html,hashlib
from decimal import Decimal
B=Path(__file__).resolve().parent
s=json.loads((B/'exploration_summary.json').read_text())
def read(name):
 with open(B/name,newline='') as f:return list(csv.DictReader(f))
inventory=read('attribute_inventory_and_screening.csv');basins=read('basin_subbasin_audit.csv');idx=read('subbasin_index.csv')

# Preserve masked numeric values separately from true seasonal categories.
cat_path=B/'categorical_attribute_values.csv';raw=read('categorical_attribute_values.csv')
masked=[r for r in raw if r['value'].strip()=='--'];cats=[r for r in raw if r['value'].strip()!='--']
if masked:
 with open(B/'masked_numeric_values.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(masked[0]));w.writeheader();w.writerows(masked)
with open(cat_path,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['basin_id','subbasin_id','catchment_split','attribute','value']);w.writeheader();w.writerows(cats)
weight_lookup={(r['basin_id'],r['subbasin_id']):float(r['pca_fit_weight']) for r in idx};cat_mass=collections.Counter();cat_counts=collections.defaultdict(collections.Counter)
for r in cats:
 if r['catchment_split']=='training_donor':cat_mass[r['attribute']]+=weight_lookup[(r['basin_id'],r['subbasin_id'])];cat_counts[r['attribute']][r['value']]+=1
for r in inventory:
 numeric_n=int(r.get('valid_numeric_training_subbasins',r['valid_training_subbasins']));cat_n=sum(cat_counts[r['attribute']].values());total=int(r['training_subbasins']);nm=float(r.get('missing_fraction_numeric_basin_area_weighted',r['missing_fraction_basin_area_weighted']))
 r['valid_numeric_training_subbasins']=numeric_n;r['valid_training_subbasins']=numeric_n+cat_n;r['nonnumeric_training_rows']=cat_n;r['training_category_counts']=json.dumps(dict(cat_counts[r['attribute']]))
 r['missing_fraction_numeric_rows']=1-numeric_n/total;r['missing_fraction_numeric_basin_area_weighted']=nm;r['missing_fraction_rows']=max(0.,1-(numeric_n+cat_n)/total);r['missing_fraction_basin_area_weighted']=max(0.,min(1.,nm-cat_mass[r['attribute']]))
 if 'conductivity_' in r['attribute'] and r['domain_role']=='categorical_season_review':
  r['domain_role']='physical_candidate';r['domain_reason']='Numeric soil conductivity; double dash is a masked/missing value, not a category.'
  r['quality_action']='review_exact_duplicate' if r['duplicate_of'] else 'exclude_constant_training' if r['constant_training']=='True' else 'review_high_missingness' if r['missing_fraction_basin_area_weighted']>.4 else 'review'
with open(B/'attribute_inventory_and_screening.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
s['categorical_attributes']=sum(bool(v) for v in cat_counts.values());s['categorical_value_records']=len(cats);s['quality_actions']=dict(collections.Counter(r['quality_action'] for r in inventory));s['domain_roles']=dict(collections.Counter(r['domain_role'] for r in inventory))
if (B/'masked_numeric_values.csv').exists():
 maskrows=read('masked_numeric_values.csv');s['masked_numeric_records']=len(maskrows);s['masked_numeric_attributes']=len({r['attribute'] for r in maskrows})
with open('/project/6102174/data/misc-data/CAMELS-SPAT/attributes/attributes-lumped.csv') as f:rr=csv.reader(f);next(rr);lnames={r[1] for r in rr}
subonly=[{k:r[k] for k in ['attribute','category','unit','source']} for r in inventory if r['attribute'] not in lnames]
with open(B/'subbasin_only_attributes.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['attribute','category','unit','source']);w.writeheader();w.writerows(subonly)
s['subbasin_only_attributes']=len(subonly);(B/'exploration_summary.json').write_text(json.dumps(s,indent=2))

# Canonical MERIT IDs are only overlap flags; clipped geometries can differ.
sets={label:{str(Decimal(r['subbasin_id']).normalize()) for r in idx if r['seen_unseen']==label} for label in ['Seen','Unseen']}
shared=sets['Seen']&sets['Unseen'];(B/'subbasin_id_overlap_audit.json').write_text(json.dumps(dict(shared_raw_merit_ids=len(shared),training_unique_raw_merit_ids=len(sets['Seen']),unseen_unique_raw_merit_ids=len(sets['Unseen']),interpretation='Shared IDs indicate possible physical overlap, not proof of identical clipped polygons. Use basin_id plus subbasin_id as the record key.'),indent=2))

pair_counts=collections.Counter()
for r in read('high_correlation_review_pairs.csv'):
 pair_counts[r['attribute_1']]+=1;pair_counts[r['attribute_2']]+=1
lookup_inventory={r['attribute']:r for r in inventory};plan=[]
for r in inventory:
 role=r['domain_role'];quality=r['quality_action'];include=role=='physical_candidate';action='candidate_retain';reason='Physical descriptor; retain subject to transformation and redundancy review.'
 if not include:action='defer_from_primary_physical_pca';reason=r['domain_reason']
 if quality in ('exclude_all_missing','exclude_constant_training'):include=False;action=quality;reason='No usable training variation for a fitted PCA.'
 if quality=='review_numerically_near_constant':include=False;action='review_numerical_precision';reason='Extremely small variation may reflect numerical precision; verify before standardizing.'
 if quality=='review_high_missingness':include=False;action='review_missing_coverage';reason='More than 40 percent missing under basin-balanced area weights; inspect geographical coverage.'
 if r['duplicate_of'] and include and lookup_inventory[r['duplicate_of']]['domain_role']=='physical_candidate':include=False;action='candidate_remove_exact_duplicate';reason='Exact training values and matching units duplicate '+r['duplicate_of']+'.'
 if quality=='review_categorical_encoding':include=False;action='encode_or_defer_categorical';reason='Observed labels are not missing numeric data and require appropriate encoding.'
 notes=r['review_flags']
 if 'fraction' in r['attribute'] and float(r['training_max'])>1.000001:notes+=';check_fraction_range_and_units'
 plan.append(dict(attribute=r['attribute'],category=r['category'],unit=r['unit'],source=r['source'],initial_physical_pca_candidate=include,recommended_action=action,reason=reason,high_correlation_review_pairs=pair_counts[r['attribute']],transform_and_domain_flags=notes))
plan_by_name={r['attribute']:r for r in plan};families=collections.defaultdict(list)
for r in inventory:
 if r['domain_role']=='physical_candidate' and r['quality_action'] in ('review','review_exact_duplicate'):
  families[r['duplicate_of'] or r['attribute']].append(r['attribute'])
for family in families.values():
 if len(family)<2:continue
 preferred=min(family,key=lambda name:(not name.endswith('_mean'),name))
 for name in family:
  item=plan_by_name[name];item['initial_physical_pca_candidate']=name==preferred;item['recommended_action']='candidate_retain' if name==preferred else 'candidate_remove_exact_duplicate';item['reason']='Training-identical family; retain '+preferred+' as the interpretable representative.'
anomalies=[]
for r in inventory:
 if 'std' in r['attribute'] and float(r['training_min']) < -1e-8:
  anomalies.append(dict(attribute=r['attribute'],source=r['source'],unit=r['unit'],training_min=r['training_min'],training_max=r['training_max'],reason='Negative value in a field named as a standard deviation; verify definition and source implementation. Do not take absolute values.'))
  plan_by_name[r['attribute']]['transform_and_domain_flags']+=';negative_standard_deviation_source_review'
with open(B/'data_quality_anomalies.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['attribute','source','unit','training_min','training_max','reason']);w.writeheader();w.writerows(anomalies)
s['negative_std_attributes_flagged']=len(anomalies);(B/'exploration_summary.json').write_text(json.dumps(s,indent=2))

with open(B/'feature_reduction_plan.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(plan[0]));w.writeheader();w.writerows(plan)
ps=dict(provisional_not_final=True,initial_physical_pca_candidates=sum(r['initial_physical_pca_candidate'] for r in plan),recommended_actions=dict(collections.Counter(r['recommended_action'] for r in plan)),pca_fitted=False,automatic_correlation_removal=False)
(B/'screening_plan_summary.json').write_text(json.dumps(ps,indent=2))

bycat=collections.Counter(r['category'] for r in inventory)
lines=['# Subbasin attribute exploration: phase 1','This phase inventories and screens candidate attributes. It does not fit PCA or retrain any LSTM. Original source attributes and model folders are unchanged.','## Verified dataset scope',f"All {s['basins']} basins inspected: 1326 training donors and 100 unseen basins. There are {s['subbasin_records']} basin-subbasin records with {s['subbasin_attributes']} attributes each. Whole-basin files contain {s['whole_basin_attributes']} attributes. The {s['whole_basin_only_attributes']} remaining fields are listed separately.",f"Subbasins per basin range from {s['subbasin_count_min']} to {s['subbasin_count_max']}. {s['single_subbasin_basins']} basins have only one subbasin. Their between-subbasin variability cannot be resolved; a future zero dispersion must be accompanied by a single-subbasin flag.",'## Attribute families','Category | Attributes','--- | ---']
lines += [f'{k} | {v}' for k,v in bycat.items()]
lines += ['## Screening results','Action | Attributes','--- | ---']+[f'{k} | {v}' for k,v in s['quality_actions'].items()]
lines += ['## Provisional reduction plan', f"{ps['initial_physical_pca_candidates']} attributes are initial physical-PCA candidates. feature_reduction_plan.csv records a proposed decision and reason for every attribute. No PCA has been fitted.", '## Domain recommendations',"- Keep raw data intact. Constant training attributes and all-missing training attributes cannot contribute to a fitted PCA. Low variance or low frequency alone does not prove hydrological irrelevance.","- Treat exact duplicates and absolute rank correlations >=0.98 as review candidates. Correlation is not a reason to remove every depth, season or spatial variability statistic. Diagnostic rank correlations use eight area-proportional samples per training basin, not the unseen basins.","- Evaluate a physical-properties PCA separately from climatic descriptors. RDRS summaries may extend beyond an experiment's training years. WorldClim climatology and mapped vegetation/land-cover products also have reference periods that need explicit documentation.","- Use basin_area for weights and document discretization. Keep centroid coordinates outside the primary physical-properties PCA. Upstream area and stream length may also reflect network position and scale.","- Do not feed aspect angles as ordinary linear degree variables. Review sine/cosine representation and the suitability of circular variability statistics.","- Review soil uncertainty estimates and no-data/unclassified land-cover fractions as data-quality indicators. Preserve physically useful soil depths and heterogeneity measures. The existing log-permeability fields are already transformed and must not be logged again.","- Land-cover fractions and soil texture compositions are constrained groups. Choose an appropriate representation before standardization/PCA. Review area/volume totals and counts for area normalization, rather than interpreting subdivision size as a catchment property.",'## Fitting and aggregation policy','Only subbasins belonging to the 1326 donors may fit missing-value treatment, transformations, scaling, feature selection and the shared PCA. Every donor gets equal total fit weight; within that donor, subbasins get their area fractions. The 100 held-out basins receive transformations only after fitting.',"Fit one shared set of loadings, not a separate PCA per basin. Evaluate 10 and 20 components using variance retention, loadings, and validation. For each basin, retain area-weighted mean and population standard deviation of each PC score, plus the granularity flag. Means represent average properties; dispersion represents between-subbasin heterogeneity, not spatial arrangement.",f"Largest relative difference between summed subbasin areas and lumped basin area: {s['area_sum_relative_error_max']:.6g}. Inspect the basin audit before aggregation. {len(shared)} raw MERIT IDs occur in both split groups; this flags possible overlap, and does not establish identical clipped geometry.",'## Files','- attribute_inventory_and_screening.csv: all attribute names, categories, sources, units, training-only missingness/ranges/weighted moments, domain roles and review flags.','- basin_subbasin_audit.csv: counts, area totals, area agreement, effective subbasin count and single-subbasin flag.','- subbasin_index.csv: parent basin, subbasin ID, Seen/Unseen, areas, within-basin weights and training-only fit weights.','- subbasin_attribute_matrix.npz: numeric matrix; categorical values are NaN until encoding. Rows follow subbasin_index.csv and attribute names give the columns.', '- categorical_attribute_values.csv: observed seasonal labels. masked_numeric_values.csv: missing soil conductivity values encoded as double dash in the source.','- exact_duplicate_candidates.csv: full-training exact value matches with matching units.','- high_correlation_review_pairs.csv: training-only sampled rank-correlation candidates.','- whole_basin_only_attributes.csv: fields absent from the subbasin schema.','- exploration_summary.json / fit_scope.json / subbasin_id_overlap_audit.json: scope, definitions and overlap flags.','- 01_Subbasin_Attribute_Exploration.ipynb: report notebook with the full audit source embedded; rerun switch defaults to False.','- explore_subbasin_attributes.py: full numerical audit, submitted as a CPU job.','## Documentation','CAMELS-SPAT README_V2.txt and known_issues.txt in the source archive. Attribute definitions and processing context: https://hess.copernicus.org/articles/29/5791/2025/ . The attribute CSV metadata supplies Category, Attribute, Unit and Source.']
lines += ['## Source-statistic anomalies', str(len(anomalies))+' fields named as standard deviations have negative training values. See data_quality_anomalies.csv. Check source definitions; do not silently repair.', 'There are '+str(s['whole_basin_only_attributes'])+' whole-basin-only names and '+str(s['subbasin_only_attributes'])+' subbasin-only names. The net count difference is not the number of unmatched names.']
(B/'EXPLORATION_REPORT.md').write_text(chr(10).join(lines)+chr(10))
(B/'README.md').write_text('Start with EXPLORATION_REPORT.md or 01_Subbasin_Attribute_Exploration.ipynb. PCA and LSTM training have not yet been performed in this folder. All screening suggestions are proposals rather than changes to original source attributes.'+chr(10))
cells=[]
def md(text):cells.append(dict(cell_type='markdown',metadata={},source=[text]))
def code(text,hidden=False):
 compile(text,'cell','exec');cells.append(dict(cell_type='code',metadata={'jupyter':{'source_hidden':True}} if hidden else {},source=[text],execution_count=None,outputs=[]))
md('# Subbasin attribute exploration'+chr(10)+f"{s['basins']} basins; {s['subbasin_records']} subbasin records; {s['subbasin_attributes']} subbasin attributes. Training-only screening; no PCA fit or LSTM training yet.")
code("from pathlib import Path\nimport csv,json,html\nfrom IPython.display import display,HTML\nB=Path.cwd()\ndef read(name):\n with open(B/name,newline='') as f:return list(csv.DictReader(f))\ndef table(rows,cols):\n text='<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c,'')))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'\n display(HTML(text))\nsummary=json.loads((B/'exploration_summary.json').read_text())\nprint(json.dumps(summary,indent=2))".replace(chr(92)+'n',chr(10)))
md('## Attribute inventory: training-only statistics')
code("inventory=read('attribute_inventory_and_screening.csv')\ntable(inventory[:25],['attribute','category','unit','source','missing_fraction_basin_area_weighted','domain_role','quality_action'])".replace(chr(92)+'n',chr(10)))
md('## Basic screening candidates')
code("flagged=[r for r in inventory if r['quality_action']!='review']\ntable(flagged[:80],['attribute','quality_action','duplicate_of','training_min','training_max'])\nprint('Flagged attributes:',len(flagged))".replace(chr(92)+'n',chr(10)))
md('## Subbasin granularity and area checks')
code("basins=read('basin_subbasin_audit.csv')\ntable(sorted(basins,key=lambda r:abs(float(r['area_relative_difference'])),reverse=True)[:20],['basin_id','catchment_split','subbasins','area_sum_km2','lumped_area_km2','area_relative_difference','effective_subbasins','single_subbasin'])".replace(chr(92)+'n',chr(10)))
md('## Highly correlated attribute pairs: review, not automatic removal')
code("pairs=read('high_correlation_review_pairs.csv')\ntable(sorted(pairs,key=lambda r:abs(float(r['spearman_r'])),reverse=True)[:40],['attribute_1','attribute_2','spearman_r','category_1','category_2'])".replace(chr(92)+'n',chr(10)))
md('## Reproduce the full audit'+chr(10)+'Run on a CPU compute session with NumPy and access to the original CAMELS-SPAT archive. The source is embedded below. RUN_FULL_AUDIT=False only reads existing reports. Setting True recomputes outputs in this experiment folder and may take several minutes.')
code('AUDIT_SOURCE='+repr((B/'explore_subbasin_attributes.py').read_text()),True)
code("RUN_FULL_AUDIT=False\nif RUN_FULL_AUDIT:\n exec(compile(AUDIT_SOURCE,'subbasin_audit','exec'))".replace(chr(92)+'n',chr(10)))
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
for i,c in enumerate(cells):c['id']='c'+str(i)
(B/'01_Subbasin_Attribute_Exploration.ipynb').write_text(json.dumps(nb))
assert len(inventory)==1101 and len(basins)==1426 and len(idx)==s['subbasin_records']
assert sum(float(r['pca_fit_weight']) for r in idx if r['seen_unseen']=='Unseen')==0
(B/'phase1_verification.json').write_text(json.dumps(dict(passed=True,attribute_rows=len(inventory),basin_rows=len(basins),subbasin_rows=len(idx),notebook_code_cells_compile=True,unseen_fit_weight_zero=True,pca_fitted=False,prior_models_modified=False),indent=2))
print('REPORT AND NOTEBOOK CREATED',flush=True)
