# Lumped Data Explorer

Two authoritative data files contain all 1,426 catchments:

- `daymet_lumped_daily.csv`: existing Daymet model records, 1980-01-01 through 2009-12-31.
- `era5_lumped_daily.csv`: extended ERA5 model daily records, 1950-01-01 through early January 2023. Coverage and missing values vary by catchment and feature.

Each row is one catchment/day. Columns: basin_id, split, date, prcp, tmax, tmin, srad, vp, dayl, pet, q_obs_mm_day. Basin identifiers remain text. Missing values are blank or NaN, never artificial zeros. No catchment attributes or LSTM predictions are included. Discharge is observed runoff depth in mm/day. Forcing units are recorded in manifest.json. ERA5 is the previously validated daily conversion used in the extended model, not the raw hourly variables. PET derivations differ between sources.

## Open or restart on Nibi

Start a CPU JupyterLab session (no GPU needed) and open `Open_Dashboard.ipynb` in this folder. Run its launch cell and click the displayed dashboard link. The authenticated Jupyter proxy makes the loopback-only server available in your browser. When the Jupyter session expires, start a new one and rerun the launch cell; the datasets remain saved.

Alternatively in that Jupyter session's terminal:

```bash
cd '/home/pollen26/Desktop/LSTM_RR_Modelling/Lumped_Data_Explorer'
'/home/pollen26/Desktop/LSTM_RR_Modelling/Model_A/.venv/bin/python' dashboard.py --port 8765
```

Open the same Jupyter session's base URL followed by `proxy/8765/`. On another computer, install NumPy and run `python dashboard.py`, then visit http://127.0.0.1:8765/ . All assets are self-contained; no CDN is required.

## Explore

Select Daymet, ERA5, or both. Search and select catchments, or select all matching the current catchment-group/search filter. Choose one, two, or three vertically stacked graphs and a parameter for each. A shared date interval controls every graph and statistics row. Use date inputs, either slider, shared/full period buttons, or drag across a graph to zoom. Hover shows the nearest actual daily observation. PNG exports save each graph. Summary headers sort ascending/descending, the row filter searches the current results, and summary export respects its filter and sort order. Summary pagination changes only the visible table page; exports include all matching rows.

Statistics are computed separately by source, basin, and displayed parameter using every valid daily value in the selected interval. Quartiles use linear interpolation. Standard deviation is sample std (ddof=1), undefined for one observation. Zero percentage uses exact zeros among valid daily values. Dates are inclusive. No pooling across different units or basins is done. Missing observations are excluded from statistics and retained as gaps in plots.

Large plot selections use min/max sampling, explicitly labelled. Hover still reports actual daily values, but only displayed points are hoverable; narrow the date range or select fewer basins for detailed inspection. Statistics are never sampled. Identical discharge traces can occur across sources because observed discharge is shared, and ERA5 has a longer record. An empty period has zero valid days and undefined statistics.

## Reproduce from CSV

The `cache/` NPY files accelerate browsing. They are derived solely from the two CSVs. `prepare_data.py` rebuilds them and verifies all basin IDs, split labels, unique ascending dates and row counts. Existing CSVs are reused. To move the dashboard, copy the two CSVs, application files, manifest.json and cache directory. For a CSV-only cache rebuild outside the original experiments, retain manifest.json and use `rebuild_cache.py`. No model training is performed.

Dataset and application checks are saved in dataset_verification.json and dashboard_verification.json. Application scripts do not modify earlier experiments.
