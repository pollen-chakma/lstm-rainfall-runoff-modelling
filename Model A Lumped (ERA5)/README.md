# Model A Lumped (ERA5)

Independent experiment; Daymet result files are preserved. The old Model_A_Unseen100 path is a symlink to Model A Lumped (Daymet).

## Fixed protocol
1326 donor catchments and the same 100 unseen IDs. Same Daymet-derived climate strata, same five attributes. Train 1980–1999; donor validation 2000–2004; both tests 2005–2009. LSTM hidden size 64, lookback 365, Adam 0.001, gradient clip 1, seed 42, 20 epochs, batch 128, 256 samples per donor per epoch, up to 128 fixed validation windows per donor. Fresh random initialization; no Daymet weights used. All preprocessing is fitted on training donor data only.

## ERA5 daily inputs
- prcp: sum(max(mtpr,0)*3600), mm/day
- tmax: max(t)-273.15, degC
- tmin: min(t)-273.15, degC
- srad: sum(max(msdwswrf,0)*3600)/astronomical_daylength_seconds, daylight-mean W/m2; zero during polar night
- vp: mean(e)*1000, Pa
- dayl: FAO-56 astronomical daylength from ERA5 basin latitude and Gregorian day-of-year, seconds
- pet: max(-sum(mper*3600),0), mm/day; native ERA5 potential evaporation, not Daymet Priestley-Taylor PET

Fluxes use the hour ending at their timestamp; instantaneous temperature and vapour pressure use the timestamp hour. All dates are local standard dates already encoded in CAMELS-SPAT. Each daily feature requires 24 valid hours. A model window requires 365 complete forcing days. Tiny negative precipitation/radiation artifacts are clipped at zero. Net daily PET below zero is clipped at zero; counts are audited.

ERA5 daily extrema are extrema of basin-average hourly temperature; Daymet temperatures are spatial averages of gridded daily extrema. Day length is approximated from ERA5 basin latitude with FAO-56 geometry. Native ERA5 PET differs from Daymet Priestley–Taylor PET. Thus this is a seven-input forcing-product comparison; these feature-definition differences must be stated in research reporting.

## Sources
- CAMELS-SPAT variable and timestamp conventions: https://hess.copernicus.org/articles/29/5791/2025/
- Daymet srad is a daylight-period mean: https://daymet.ornl.gov/overview
- FAO-56 day length geometry: https://www.fao.org/4/x0490e/x0490e07.htm
- ECMWF surface-to-atmosphere flux sign convention: https://confluence.ecmwf.int/pages/viewpage.action?pageId=669811810

Shared polygon overlap is unchanged from Daymet. The 100 holdout IDs are unseen during training but not all are geometrically independent of donor drainage areas.
