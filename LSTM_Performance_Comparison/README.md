# Basin LSTM performance comparison

One basin per row. Catchment IDs are joined by exact text ID, never by row position. All 1,426 IDs are retained (1,326 training donors, 100 unseen catchments). Scores for donors are held-out temporal test scores, not training-fit scores.

## Models and periods

Daymet_MetAttr_Test2005_2009: Model A Lumped (Daymet). Training 1980-1999; validation 2000-2004; testing 2005-2009. Seed 42.

Sources: Model A Lumped (Daymet)/run_seed42/temporal_test_2005_2009/metrics.csv and Model A Lumped (Daymet)/run_seed42/unseen_metrics.csv.

Daymet_MetOnly_Test2005_2009: Model A Meteorology Only (Daymet). Training 1980-1999; validation 2000-2004; testing 2005-2009. Seed 42.

Sources: Model A Meteorology Only (Daymet)/results_seed42/donor_temporal/metrics.csv and Model A Meteorology Only (Daymet)/results_seed42/unseen100/metrics.csv.

ERA5_MetAttr_Test2005_2009: Model A Lumped (ERA5). Training 1980-1999; validation 2000-2004; testing 2005-2009. Seed 42.

Sources: Model A Lumped (ERA5)/run_seed42/temporal_test_2005_2009/metrics.csv and Model A Lumped (ERA5)/run_seed42/unseen_metrics.csv.

ERA5_MetAttr_Test2015_2023: Model A Lumped (ERA5, 1950-2023). Training 1950-2009; validation 2010-2014; testing 2015-early 2023. Seed 42.

Sources: Model A Lumped (ERA5, 1950-2023)/run_seed42/temporal_test_2015_latest/metrics.csv and Model A Lumped (ERA5, 1950-2023)/run_seed42/unseen_metrics.csv.

The extended ERA5 model uses a different training and test period. Its score differences cannot be attributed solely to the meteorological source or attribute inputs. Its requested end_exclusive is 2023-01-04; forcing/observation availability determines the actual scored days for each basin.

## Columns

NSE and KGE: original saved metrics, ideal value 1. Bias_percent: 100*(sum(predicted)-sum(observed))/sum(observed), positive means overprediction, ideal 0. N_scored_days: number of paired valid observation/prediction days. Status preserves source explanations. Blank metrics are unavailable/undefined, never zero-filled. Counts of zero are retained when confirmed by a source metric/coverage audit.

catchment_split is training_donor or unseen_test. basin_group and climate_class come from the original split manifest. Climate labels are meteorological rain/snow proxies, not formal climate zones. The existing spatial-overlap limitation remains: 26 unseen IDs share polygon IDs with donors.

ERA5 meteorology-only has not yet been trained and is not represented by fabricated scores.

The CSV preserves source numeric precision. comparison_verification.json records coverage and source hashes.

seen_unseen is the second column: Seen = training_donor (1,326 basins), Unseen = unseen_test (100 basins). Existing row order and scores are unchanged.
