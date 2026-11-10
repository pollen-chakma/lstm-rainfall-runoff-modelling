# Final subbasin PCA inputs

Completed on Nibi. One shared PCA was fitted only to the 30,604 subbasin records of the 1,326 donor basins, then applied unchanged to all 32,802 records belonging to 1,426 basins. The 100 held-out basins contribute no fitting weight. No discharge data were used.

## Files for model input
- basin_pca10_mean_std_features.csv: 1,426 rows; 20 PC features (10 means + 10 standard deviations), plus five identifying/flag columns.
- basin_pca20_mean_std_features.csv: 1,426 rows; 40 PC features (20 means + 20 standard deviations), plus the same five columns.
- single_subbasin_flag is an additional recommended model input; 304 basins have one subbasin and therefore zero between-subbasin standard deviations. This does not imply no intrinsic heterogeneity.
- basin_id is the join key. seen_unseen, catchment_split and n_subbasins are metadata, not default model predictors. Read IDs as strings.

## Method
The screening starts from 1,101 subbasin attributes (not the 1,178 whole-basin attributes). The physical-property shortlist contained 317 candidates. Final PCA uses 312 attributes: {'Land cover': 72, 'Topography': 12, 'Soil': 206, 'Open water': 14, 'Geology': 8}.
Climate, hydrological response, identifiers, raw circular aspect, quality/uncertainty fields and network upstream area are excluded. Counts and extensive lake/reservoir/stream measures are converted to area-normalized measures. Fraction fields use a square-root transformation; sufficiently skewed fields use a training-derived log1p or signed asinh transformation. Invalid values become missing and are imputed with donor basin-area-weighted medians. Full per-attribute details are in final_pca/final_preprocessing.csv and final_attribute_decisions.csv.
Redundancy removal uses donor-only weighted rank correlation >=0.98 only within the same min/mean/max property, source and soil depth; useful standard deviations and separate depths are preserved.
Each donor basin has equal total fitting weight, distributed among its subbasins by area. Features are standardized using these weights, then divided by the square root of the number of retained features in their physical domain. This gives each domain equal initial total variance. Consequently explained variance refers to this transformed, domain-balanced space.
The 10-PC set explains 69.47% and the 20-PC set 84.41%. The first 10 PCs are identical between files.
For each basin, PC_mean=sum(area_fraction*PC_score), and PC_std=sqrt(sum(area_fraction*(PC_score-PC_mean)^2)); this is a population standard deviation, not an uncertainty estimate. These summaries capture average properties and dispersion, but do not encode spatial arrangement or routing.

## Reuse and reproducibility
Join one of these files to daily forcing/discharge by basin_id. Use the PC columns and optionally single_subbasin_flag as static LSTM inputs. Fit any subsequent LSTM static-feature scaler using donor basins only. Keep the existing donor/unseen IDs and chronological splits. Do not refit PCA on test basins.
final_pca/model_input_columns_pca10.json and pca20.json list exact model and metadata columns. final_pca/pca_transform.npz stores the fixed numerical transform; pca_loadings.csv gives interpretation and explained_variance.csv gives every component. subbasin_pc_scores.npz follows subbasin_index.csv row order.
fit_final_pca.py is the complete fitting/export source. It refuses to overwrite existing final CSVs. For a fresh recomputation, copy the source matrix, index, basin audit, reduction plan, inventory and script into a separate directory, then run the script with NumPy installed. The notebook 02_Final_Subbasin_PCA.ipynb displays and explains the completed outputs without refitting by default.
Verification includes finite values, all basin IDs, 1,326/100 split preservation, zero unseen fitting weight, orthogonality and weighted score covariance, area-weighted variance identities, shared first 10 PCs, and independent reapplication of the saved transformation and aggregation to all rows.

## Interpretation limits
Physical source maps represent their own reference dates; this is a retrospective static-property experiment. Existing audit reports 440 raw subbasin IDs shared across donor and held-out basins; these can represent nested or overlapping geography, so the held-out set is unseen by basin ID but not guaranteed spatially disjoint. The original split has been preserved. Composite basin_id/subbasin_id identifies records. The 12 anomalous RDRS temperature standard-deviation fields found in exploration are outside this physical PCA.
No new LSTM has been trained in this step. These files are static inputs for the next model experiment.
