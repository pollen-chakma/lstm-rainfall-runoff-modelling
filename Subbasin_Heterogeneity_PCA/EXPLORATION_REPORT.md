# Subbasin attribute exploration: phase 1
This phase inventories and screens candidate attributes. It does not fit PCA or retrain any LSTM. Original source attributes and model folders are unchanged.
## Verified dataset scope
All 1426 basins inspected: 1326 training donors and 100 unseen basins. There are 32802 basin-subbasin records with 1101 attributes each. Whole-basin files contain 1178 attributes. The 80 remaining fields are listed separately.
Subbasins per basin range from 1 to 227. 304 basins have only one subbasin. Their between-subbasin variability cannot be resolved; a future zero dispersion must be accompanied by a single-subbasin flag.
## Attribute families
Category | Attributes
--- | ---
Climate | 625
Land cover | 74
Topography | 20
Soil | 352
Open water | 22
Geology | 8
## Screening results
Action | Attributes
--- | ---
review | 1076
review_exact_duplicate | 16
review_categorical_encoding | 4
exclude_constant_training | 5
## Provisional reduction plan
317 attributes are initial physical-PCA candidates. feature_reduction_plan.csv records a proposed decision and reason for every attribute. No PCA has been fitted.
## Domain recommendations
- Keep raw data intact. Constant training attributes and all-missing training attributes cannot contribute to a fitted PCA. Low variance or low frequency alone does not prove hydrological irrelevance.
- Treat exact duplicates and absolute rank correlations >=0.98 as review candidates. Correlation is not a reason to remove every depth, season or spatial variability statistic. Diagnostic rank correlations use eight area-proportional samples per training basin, not the unseen basins.
- Evaluate a physical-properties PCA separately from climatic descriptors. RDRS summaries may extend beyond an experiment's training years. WorldClim climatology and mapped vegetation/land-cover products also have reference periods that need explicit documentation.
- Use basin_area for weights and document discretization. Keep centroid coordinates outside the primary physical-properties PCA. Upstream area and stream length may also reflect network position and scale.
- Do not feed aspect angles as ordinary linear degree variables. Review sine/cosine representation and the suitability of circular variability statistics.
- Review soil uncertainty estimates and no-data/unclassified land-cover fractions as data-quality indicators. Preserve physically useful soil depths and heterogeneity measures. The existing log-permeability fields are already transformed and must not be logged again.
- Land-cover fractions and soil texture compositions are constrained groups. Choose an appropriate representation before standardization/PCA. Review area/volume totals and counts for area normalization, rather than interpreting subdivision size as a catchment property.
## Fitting and aggregation policy
Only subbasins belonging to the 1326 donors may fit missing-value treatment, transformations, scaling, feature selection and the shared PCA. Every donor gets equal total fit weight; within that donor, subbasins get their area fractions. The 100 held-out basins receive transformations only after fitting.
Fit one shared set of loadings, not a separate PCA per basin. Evaluate 10 and 20 components using variance retention, loadings, and validation. For each basin, retain area-weighted mean and population standard deviation of each PC score, plus the granularity flag. Means represent average properties; dispersion represents between-subbasin heterogeneity, not spatial arrangement.
Largest relative difference between summed subbasin areas and lumped basin area: 2.67712e-07. Inspect the basin audit before aggregation. 440 raw MERIT IDs occur in both split groups; this flags possible overlap, and does not establish identical clipped geometry.
## Files
- attribute_inventory_and_screening.csv: all attribute names, categories, sources, units, training-only missingness/ranges/weighted moments, domain roles and review flags.
- basin_subbasin_audit.csv: counts, area totals, area agreement, effective subbasin count and single-subbasin flag.
- subbasin_index.csv: parent basin, subbasin ID, Seen/Unseen, areas, within-basin weights and training-only fit weights.
- subbasin_attribute_matrix.npz: numeric matrix; categorical values are NaN until encoding. Rows follow subbasin_index.csv and attribute names give the columns.
- categorical_attribute_values.csv: observed seasonal labels. masked_numeric_values.csv: missing soil conductivity values encoded as double dash in the source.
- exact_duplicate_candidates.csv: full-training exact value matches with matching units.
- high_correlation_review_pairs.csv: training-only sampled rank-correlation candidates.
- whole_basin_only_attributes.csv: fields absent from the subbasin schema.
- exploration_summary.json / fit_scope.json / subbasin_id_overlap_audit.json: scope, definitions and overlap flags.
- 01_Subbasin_Attribute_Exploration.ipynb: report notebook with the full audit source embedded; rerun switch defaults to False.
- explore_subbasin_attributes.py: full numerical audit, submitted as a CPU job.
## Documentation
CAMELS-SPAT README_V2.txt and known_issues.txt in the source archive. Attribute definitions and processing context: https://hess.copernicus.org/articles/29/5791/2025/ . The attribute CSV metadata supplies Category, Attribute, Unit and Source.
## Source-statistic anomalies
12 fields named as standard deviations have negative training values. See data_quality_anomalies.csv. Check source definitions; do not silently repair.
There are 80 whole-basin-only names and 3 subbasin-only names. The net count difference is not the number of unmatched names.


## Before fitting PCA

These checks assess usability and redundancy, not predictive significance. The provisional physical shortlist contains 208 soil, 64 land-cover, 11 topographic, 20 open-water and 8 geology attributes. Review domain-block balancing or a smaller physically representative soil set before a shared PCA, so the number of soil descriptors does not set its influence by default. Resolve transformation flags and uncertainty/reference-period choices before fitting.
