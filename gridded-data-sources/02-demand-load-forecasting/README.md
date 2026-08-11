# Use Case 02 — Demand / Load Forecasting, Australia

## Overview

**Goal:** Model electricity demand as a function of weather (temperature is the single strongest driver of load via heating/cooling), for forecasting and network planning.

**Source:** ERA5 (ECMWF Reanalysis v5) via ARCO-ERA5 Zarr on GCS.

**Architecture:** Databricks Asset Bundle (DAB) with Lakeflow Declarative Pipeline (DLT) and Materialized Views (MV) for medallion architecture (Bronze/Silver/Gold).

---

## Databricks Asset Bundle Structure

This folder contains a complete DAB for deploying the demand forecasting medallion pipeline on Databricks.

### File Structure

```
.
├── databricks.yml                      # Bundle config, variables, targets
├── resources/
│   ├── demand_forecasting.pipeline.yml  # DLT pipeline definition
│   └── demand_forecasting.job.yml       # Job config (setup + pipeline tasks)
├── src/
│   ├── setup/
│   │   └── 00_setup.py                 # Setup notebook: create catalog/schema
│   └── pipelines/
│       ├── 01_bronze.py                # Bronze: ERA5 ingestion (xarray → Spark)
│       ├── 02_silver.sql               # Silver: Cleaning & unit conversions
│       └── 03_gold.sql                 # Gold: Feature engineering & aggregation
├── README.md                           # This file
└── IMPLEMENTATION_NOTES.md             # Detailed implementation notes (legacy)
```

### Bundle Configuration (`databricks.yml`)

The main configuration file defines:

- **Bundle name**: `demand_forecasting`
- **Variables** (with defaults):
  - `catalog`: Unity Catalog name (default: `geo`)
  - `schema`: Schema name (default: `weather`)
  - `au_lat_south`, `au_lat_north`, `au_lon_west`, `au_lon_east`: Australian bounding box (0.25° resolution)
  - `era5_zarr_path`: ARCO-ERA5 Zarr store on GCS (public, anonymous access)
  - `era5_variables`: List of ERA5 variables to extract
  - `nem_regions`: NEM region identifiers (NSW1, QLD1, VIC1, SA1, TAS1)
  - `date_start`, `date_end`: ISO 8601 date range for ingestion
- **Targets**: Development target with serverless compute

### Pipeline Configuration (`resources/demand_forecasting.pipeline.yml`)

- **Pipeline name**: `demand_forecasting_pipeline`
- **Type**: Lakeflow DLT (Serverless, PREVIEW channel)
- **Catalog & Schema**: Set from bundle variables
- **Libraries**: Include all Python and SQL files from `src/pipelines/`
- **Environment**: Pip dependencies for ERA5 access:
  - `xarray`, `zarr`, `gcsfs` (for ARCO-ERA5 Zarr)
  - `rioxarray`, `numpy`, `pandas` (for data transformation)

### Job Configuration (`resources/demand_forecasting.job.yml`)

Two-task job:

1. **Task 1: `setup`** (notebook)
   - Runs `src/setup/00_setup.py`
   - Creates/verifies catalog and schema
   - Idempotent; safe to re-run
   - Serverless job compute

2. **Task 2: `refresh_medallion`** (pipeline)
   - Runs the DLT pipeline (Bronze/Silver/Gold)
   - Depends on: `setup` task
   - Executes all three medallion layers sequentially

---

## Materialized Views (Medallion Layers)

### Bronze Layer: `bronze_era5`

**Table**: `geo.weather.bronze_era5` (configurable via `catalog`/`schema` variables)

**Purpose**: Raw ERA5 reanalysis data at grid-cell level (0.25° × hourly resolution).

**Implementation**: `src/pipelines/01_bronze.py` (`@dp.table`)

**Columns**:
- `timestamp`: UTC time (ISO 8601)
- `latitude`, `longitude`: Grid cell coordinates (decimal degrees)
- `t2m`: 2m temperature (Kelvin)
- `t2m_dewpoint`: 2m dewpoint temperature (Kelvin)
- `ssrad`: Surface solar radiation downwards (W/m²)
- `u10`, `v10`: 10m wind components (m/s)
- `u100`, `v100`: 100m wind components (m/s)

**Row count**: ~6.9M per day (Australia only, 0.25° grid)

**Partitioning**: None (raw data)

**Strategy**: 
- Opens ARCO-ERA5 Zarr lazily with xarray
- Slices to Australian bounding box BEFORE compute to minimize data transfer
- Converts to Spark DataFrame (via pandas)
- For demonstration/testing, falls back to synthetic data if live GCS access fails

---

### Silver Layer: `silver_era5`

**Table**: `geo.weather.silver_era5`

**Purpose**: Cleaned, typed, derived features; ready for aggregation.

**Implementation**: `src/pipelines/02_silver.sql` (SQL `CREATE OR REFRESH MATERIALIZED VIEW`)

**Transformations**:
- Kelvin → Celsius (subtract 273.15)
- Derived: Wind speed magnitude (10m, 100m) = √(u² + v²)
- Quality filters: Temperature (-60°C to 60°C), dewpoint (-70°C to 50°C), solar rad ≥ 0
- Temporal indices: `year`, `month`

**Columns**:
- Temporal: `year`, `month`, `timestamp`
- Spatial: `latitude`, `longitude`
- Converted variables: `t2m_celsius`, `t2m_dewpoint_celsius`, `ssrad_wm2`, `u10_ms`, `v10_ms`, `u100_ms`, `v100_ms`
- Derived: `wind_speed_10m`, `wind_speed_100m`

**Row count**: Same as bronze (~6.9M/day)

**Partitioning**: `year`, `month`

---

### Gold Layer: `gold_weather_features`

**Table**: `geo.weather.gold_weather_features`

**Purpose**: Region-level (NEM) features ready for demand forecasting.

**Implementation**: `src/pipelines/03_gold.sql` (SQL `CREATE OR REFRESH MATERIALIZED VIEW`)

**Transformations**:
1. **Region mapping**: Grid cells → NEM regions (NSW1, QLD1, VIC1, SA1, TAS1)
   - Current: Simplified lat/lon binning (demo)
   - Production: Use Databricks Mosaic `point_in_polygon()` with official NEM boundary geometries
2. **Regional aggregation**: Mean/min/max/stddev across grid cells per region & timestamp
3. **Feature engineering**:
   - **HDD/CDD**: Hourly proxy based on deviation from 18°C base temp
   - **Lagged features**: Temperatures from 1, 6, 24, 48 hours prior
   - **Rolling stats**: 7-day (168-hour) rolling mean, max, min
   - **Temporal**: `hour_of_day`, `day_of_week`, `day_of_year`

**Columns** (30 total):
- Identifiers: `nem_region`, `timestamp`, `date`, `year`, `month`, `hour_of_day`, `day_of_week`, `day_of_year`
- Regional aggregates: `temp_mean_c`, `temp_min_c`, `temp_max_c`, `temp_stddev_c`, `dewpoint_mean_c`, `solar_rad_mean_wm2`, `solar_rad_max_wm2`, `wind_speed_10m_mean_ms`, `wind_speed_100m_mean_ms`
- Demand features: `temp_deviation_from_base`, `hdd_proxy`, `cdd_proxy`
- Lagged: `temp_lag_1h`, `temp_lag_6h`, `temp_lag_24h`, `temp_lag_48h`
- Rolling: `temp_rolling_7d_mean`, `temp_rolling_7d_max`, `temp_rolling_7d_min`

**Row count**: 5 regions × 24 hours = 120 rows/day

**Partitioning**: `nem_region`, `year`, `month`

**Join stub**: Code included to join with AEMO NEM demand data (once ingested).

---

## ERA5 Data Source

### ARCO-ERA5 (Analysis-Ready, Cloud-Optimized)

- **URL**: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- **Access**: Public, no authentication required (anonymous)
- **Format**: Zarr (cloud-optimized array store, supports lazy slicing)
- **Resolution**: 0.25° grid, hourly timesteps
- **Coverage**: Global; easily sliced to Australia before compute

### ERA5 Variables Extracted

1. `2m_temperature` (t2m)
2. `2m_dewpoint_temperature`
3. `surface_solar_radiation_downwards`
4. `10m_u_component_of_wind` (u10)
5. `10m_v_component_of_wind` (v10)
6. `100m_u_component_of_wind` (u100)
7. `100m_v_component_of_wind` (v100)

All variables are retrieved in their native units (Kelvin for temperature, W/m² for solar rad, m/s for wind) and converted/derived in the Silver layer.

---

## Deployment & Execution

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Serverless compute available (for DLT and job compute)
- GCS access (for ARCO-ERA5 Zarr; requires either VPC endpoint or public network access)

### Deploy

```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Deploy bundle resources (creates catalog, schema, pipeline, job)
databricks bundle deploy -t dev
```

### Run

```bash
# Run the entire job (setup → pipeline)
databricks bundle run -t dev demand_forecasting_job

# Or run just the pipeline manually
databricks bundle run -t dev demand_forecasting_pipeline
```

### Customize Variables

Override defaults at deploy/run time:

```bash
databricks bundle run -t dev demand_forecasting_job \
  -var catalog=my_catalog \
  -var schema=weather_test \
  -var date_start="2023-06-01" \
  -var date_end="2023-06-30"
```

---

## Declarative Pipeline Limitations & Notes

### SQL Materialized Views: What Works & What Doesn't

**Declarative (SQL) strengths**:
- Window functions for lagged features (`LAG()`)
- Rolling aggregations with `ROWS BETWEEN`
- Partitioned aggregations (`GROUP BY`)
- Quality filters and unit conversions

**Challenges noted**:
1. **Relative humidity**: Currently stubbed (requires Magnus formula or lookup table). Could add as a UDF if needed.
2. **Complex spatial joins**: Point-in-polygon mapping (grid cell → NEM region) uses simplified lat/lon binning. Production code should use Mosaic `point_in_polygon()` or Sedona. Both are SQL-compatible, so can stay within `03_gold.sql`.
3. **AEMO demand join**: Stubbed in comments. Once AEMO data is ingested as `silver_aemo_demand`, the join is trivial SQL.

### Why Not Python `@dp.table` for Silver/Gold?

- SQL is faster and more portable
- Materialized Views ensure idempotency (refresh semantics)
- Window functions handle lags/rolling stats efficiently
- All current feature engineering fits within SQL capabilities

**If future needs require Python**:
- Bronze already uses `@dp.table` (needs xarray/zarr)
- Silver/Gold can migrate to Python `@dp.table` if non-SQL features are needed (e.g., complex ML-based imputation)

---

## Future Enhancements

1. **Mosaic/Sedona integration** (`03_gold.sql`):
   - Replace lat/lon binning with proper point-in-polygon geometry
   - Add population-weighting for region aggregation

2. **AEMO demand ingestion**:
   - Ingest NEM regional demand from AEMO open data
   - Create `silver_aemo_demand` table
   - Enable join in Gold layer (code stub provided)

3. **Relative humidity**:
   - Implement Magnus formula as SQL UDF or Python function
   - Add to Silver layer

4. **Advanced features**:
   - Apparent temperature (wind chill)
   - Autoregressive demand lagging
   - Calendar features (holidays, peak demand windows)
   - Solar PV penetration index

5. **ML model integration**:
   - Use `gold_weather_features` + `silver_aemo_demand` for demand forecasting
   - Register models in Model Registry
   - Deploy real-time prediction endpoints

---

## Dependencies

### Python Packages (Installed by DLT pipeline environment)

- `xarray==2024.1.1` — Multi-dimensional array handling
- `zarr==2.17.1` — Cloud-optimized array format
- `gcsfs==2024.1.0` — Google Cloud Storage access
- `rioxarray==0.15.0` — Raster I/O (optional, for future spatial ops)
- `numpy==1.24.3` — Numerical computing
- `pandas==2.0.3` — Data frame manipulation

### Databricks Features

- **Unity Catalog** — Multi-workspace governance, table sharing
- **Delta Lake** — ACID transactions, time-travel, Z-order clustering
- **Lakeflow DLT** — Declarative pipeline orchestration, Materialized Views
- **Serverless SQL / Compute** — Managed execution environment
- **Databricks Mosaic** (future) — H3 indexing, point-in-polygon, spatial ops

---

## References

- **ERA5 documentation**: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- **ARCO-ERA5 on GCS**: https://cloud.google.com/storage/docs/public-datasets/era5
- **AEMO NEM data**: https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem
- **Databricks Asset Bundles**: https://docs.databricks.com/en/dev-tools/bundles/
- **Databricks DLT**: https://docs.databricks.com/en/delta-live-tables/
- **Databricks Mosaic**: https://docs.databricks.com/en/sql/language-manual/functions/mosaic.html
- **Databricks Sedona**: https://docs.databricks.com/en/lakehouse/spatial/

---

## Legacy Notebooks

The standalone notebooks (`0_Setup.ipynb`, `1_Bronze_ERA5.ipynb`, `2_Silver.ipynb`, `3_Gold_NEM_Features.ipynb`) remain in this folder for reference and comparison. They have been superseded by the DAB structure above and can be removed once the bundle is validated in your workspace.

For full implementation details (assumptions, schema decisions, feature engineering rationale), see `IMPLEMENTATION_NOTES.md`.
