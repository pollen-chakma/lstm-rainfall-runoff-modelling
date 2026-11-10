# Poor-basin performance findings

Exploratory associations only. Input: basin_diagnostic_dataset.csv.
NSE and KGE flags are analyzed separately. Extended ERA5 has a different test period.

## PCA10: Seen
NSE negative: 84/1317; KGE negative: 111/1317; KGE below mean-flow reference: 59.

- test_q_std_mm_day: negative-score median 0.1543, nonnegative-score median 1.884; rank-biserial -0.791, BH q=2.73e-32.
- test_q_max_mm_day: negative-score median 2.148, nonnegative-score median 21.6; rank-biserial -0.783, BH q=5.79e-32.
- train_q_std_mm_day: negative-score median 0.2574, nonnegative-score median 1.979; rank-biserial -0.726, BH q=1.11e-27.
- test_q_mean_mm_day: negative-score median 0.06798, nonnegative-score median 1.109; rank-biserial -0.683, BH q=1.05e-24.
- test_aridity_pet_over_prcp: negative-score median 1.408, nonnegative-score median 0.7789; rank-biserial 0.678, BH q=2.15e-24.
- train_q_mean_mm_day: negative-score median 0.08259, nonnegative-score median 1.087; rank-biserial -0.668, BH q=8.86e-24.
- test_runoff_ratio: negative-score median 0.04137, nonnegative-score median 0.3938; rank-biserial -0.627, BH q=3.76e-21.
- test_prcp_mean: negative-score median 1.36, nonnegative-score median 2.901; rank-biserial -0.613, BH q=2.91e-20.

### Example worst NSE basin IDs
- CAN_05CK005: NSE -95.068; KGE -63.866; mean Q 0.0008026 mm/day; SD 0.007058; zero-flow fraction 0.758.
- CAN_05GA012: NSE -73.834; KGE -52.513; mean Q 0.001292 mm/day; SD 0.008636; zero-flow fraction 0.736.
- CAN_05GA008: NSE -17.536; KGE -19.915; mean Q 0.003109 mm/day; SD 0.0163; zero-flow fraction 0.656.
- USA_09306242: NSE -13.895; KGE -4.083; mean Q 0.01886 mm/day; SD 0.03188; zero-flow fraction 0.000.
- USA_06477500: NSE -11.731; KGE -2.203; mean Q 0.03516 mm/day; SD 0.1684; zero-flow fraction 0.067.
- CAN_05AH050: NSE -9.829; KGE -13.558; mean Q 0.005219 mm/day; SD 0.02768; zero-flow fraction 0.755.
- USA_06409000: NSE -7.991; KGE -0.805; mean Q 0.1079 mm/day; SD 0.04645; zero-flow fraction 0.000.
- USA_06408700: NSE -7.456; KGE -0.038; mean Q 0.4544 mm/day; SD 0.07453; zero-flow fraction 0.000.

## PCA10: Unseen
NSE negative: 8/100; KGE negative: 14/100; KGE below mean-flow reference: 5.

- No continuous-feature contrasts pass the exploratory BH threshold with the minimum group size; do not interpret this as evidence of no association.

### Example worst NSE basin IDs
- USA_06876700: NSE -3.093; KGE -1.344; mean Q 0.09119 mm/day; SD 0.2365; zero-flow fraction 0.000.
- USA_06404000: NSE -1.187; KGE -2.116; mean Q 0.07424 mm/day; SD 0.2189; zero-flow fraction 0.106.
- CAN_07CA012: NSE -0.675; KGE -0.335; mean Q 0.5028 mm/day; SD 0.4843; zero-flow fraction 0.000.
- CAN_06AB001: NSE -0.657; KGE -0.369; mean Q 0.2649 mm/day; SD 0.2679; zero-flow fraction 0.000.
- CAN_07HA005: NSE -0.619; KGE 0.156; mean Q 0.1993 mm/day; SD 0.3807; zero-flow fraction 0.015.
- CAN_07MA003: NSE -0.393; KGE 0.073; mean Q 0.4895 mm/day; SD 0.268; zero-flow fraction 0.000.
- CAN_07GD001: NSE -0.270; KGE -0.136; mean Q 0.1337 mm/day; SD 0.3302; zero-flow fraction 0.104.
- CAN_01FB003: NSE -0.023; KGE 0.362; mean Q 2.73 mm/day; SD 1.555; zero-flow fraction 0.000.

## Limits and next steps
- Test-flow SD and mean are mathematically involved in efficiency scores; these are diagnostics, not causal discoveries.
- Class thresholds are study-defined proxies. Missing regulation, abstraction, rating-curve and carbonate/karst evidence limits physical attribution.
- Seen/Unseen, forcing source and testing period are analyzed separately. Geographic/nested-basin dependence and one-seed model uncertainty remain.
- Confirm patterns across seeds, years and matched climate/region subsets before using them as general research conclusions.
- Do not remove basins solely because their performance is poor; investigate the observations, forcing and model limitations.