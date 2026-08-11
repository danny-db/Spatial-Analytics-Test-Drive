# Multi-Dimensional Gridded Data for Energy & Utilities (Australia)

Multi-dimensional gridded data = values on a lat/lon grid with extra dimensions
(time, height/pressure level, variable). Stored as **NetCDF, GRIB2, HDF5, or Zarr**.
This is the backbone of most energy & utility analytics: renewable resource
assessment, temperature-driven demand forecasting, and weather-driven grid risk.

Each subfolder is a self-contained use case with an Australia-focused data source,
access instructions, and Databricks ingestion notes.

| # | Use case | Primary source(s) | Format | AU coverage |
|---|----------|-------------------|--------|-------------|
| [01](./01-renewable-generation/) | Renewable generation (solar + wind) | NASA POWER, NSRDB (Himawari), WIND Toolkit* | NetCDF / HDF5 / API | Full (solar); wind needs AU substitute |
| [02](./02-demand-load-forecasting/) | Demand / load forecasting | ERA5 (ARCO-Zarr) | Zarr / NetCDF / GRIB | Full |
| [03](./03-grid-weather-risk/) | Grid / weather risk | NOAA GFS | GRIB2 | Full |

\* WIND Toolkit is US-only; see folder 01 for the Australia-relevant substitute
(Global Wind Atlas + AEMO actuals).

## Why these formats matter on Databricks

- Prefer **Zarr / Cloud-Optimized GeoTIFF (COG)** mirrors over raw NetCDF/GRIB —
  they chunk cleanly for distributed/parallel reads from object storage.
- Reading: `xarray` + `zarr` / `cfgrib` / `rioxarray`, then flatten to a Spark
  DataFrame (one row per grid-cell × timestamp) for the Silver layer.
- Spatial joins back to assets (substations, feeders, service territories,
  wind/solar farms): **Databricks Mosaic** or **Apache Sedona** (H3 or point-in-polygon).

## Australia-specific companion datasets (the "target" variables)

Gridded weather is the *feature* data. Pair it with these AU *label*/asset sources:

- **AEMO** (Australian Energy Market Operator) — NEM regional demand, dispatch,
  rooftop PV & wind generation actuals. https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem
- **Bureau of Meteorology (BOM)** — observations, warnings, radar, Himawari.
- **Geoscience Australia** — grid infrastructure & administrative boundaries.
- **AREMI** (nationalmap.gov.au/renewables) — renewable energy infrastructure layers.
