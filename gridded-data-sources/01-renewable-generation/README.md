# Use Case 01 — Renewable Generation (Solar + Wind), Australia

**Goal:** Assess and model solar & wind resource on a grid to support siting, generation forecasting, and capacity-factor analysis.

**Refactoring:** Converted from standalone notebooks to a **Databricks Asset Bundle (DAB)** with **Lakeflow Declarative Pipeline (DLT)** and **Materialized Views** (bronze/silver/gold medallion architecture).

---

## Bundle Structure

```
01-renewable-generation/
├── databricks.yml                           # DAB configuration
├── resources/
│   ├── renewable_generation.job.yml         # Job with setup + pipeline tasks
│   └── renewable_generation.pipeline.yml    # DLT pipeline definition
├── src/
│   ├── setup/
│   │   └── 00_setup.py                      # Setup notebook (catalog, schema, secrets, verification)
│   └── pipelines/
│       ├── 01_bronze.py                     # Bronze layer (NASA POWER + NREL NSRDB API ingestion)
│       ├── 02_silver.sql                    # Silver layer (cleaning, normalization, long format)
│       └── 03_gold.sql                      # Gold layer (capacity factor, aggregation)
└── README.md                                # This file
```

---

## Medallion Architecture: Bronze → Silver → Gold

### Bronze Layer (`01_bronze.py`)

Two materialized views ingesting raw data from external APIs:

| Table | Source | Variables | Coverage | Resolution | Authentication |
|-------|--------|-----------|----------|------------|-----------------|
| `bronze_nasa_power_solar` | NASA POWER API | GHI, DNI, DHI, T2M, WS10M | Global (Australia: lat -44…-10, lon 112…154) | ~0.5° × 0.625° (MERRA-2) | None (public API) |
| `bronze_nsrdb_himawari` | NREL NSRDB Himawari-8/9 | GHI, DNI, DHI, Temperature, Wind Speed | Asia-Pacific (all of Australia), ~2 km | ~2 km | NREL API key (free) |

**Data source URLs:**
- NASA POWER: `https://power.larc.nasa.gov/api/temporal/hourly/point`
- NREL NSRDB: `https://nsrdb.nrel.gov/api/v2/solar/psm3-download`

**Australian grid points** (representative; expand for full coverage):
- Sydney, Melbourne, Brisbane, Perth, Adelaide, Darwin, Hobart, Canberra

### Silver Layer (`02_silver.sql`)

One materialized view:

| Table | Purpose | Schema |
|-------|---------|--------|
| `silver_solar_irradiance` | Cleaned, unified solar irradiance (long format) | longitude, latitude, timestamp, year, month, day, location_name, data_source, variable, value, unit, ingestion_timestamp |

**Key transformations:**
- Normalize column names (ALLSKY_SFC_SW_DWN → ghi, etc.)
- Convert types (strings to doubles/timestamps)
- Pivot from wide to long format: one row per (lat, lon, timestamp, variable)
- Quality filters:
  - Irradiance (GHI/DNI/DHI): 0–1500 W/m²
  - Temperature: 250–330 K (~-23…57°C)
  - Wind speed: ≥ 0 m/s
  - Timestamps: not null
- Partitioning: by year, month, day

### Gold Layer (`03_gold.sql`)

One materialized view:

| Table | Purpose | Aggregation Level |
|-------|---------|-------------------|
| `gold_solar_wind_capacity_factor` | Solar capacity factor + wind metrics for dashboards, forecasting | Daily by location & data source |

**Schema:**
```
longitude, latitude, year, month, day, location_name, data_source,
-- Solar metrics (daily)
ghi_daily_avg, ghi_daily_peak, temperature_c_daily_avg,
solar_cf_daily_avg, solar_cf_daily_peak,
-- Wind metrics (daily)
wind_speed_daily_avg, wind_speed_daily_max, wind_speed_stddev,
wind_power_density_avg, wind_power_density_max,
-- Metadata & placeholders
model_version, hourly_record_count, last_updated,
h3_cell_id (null), asset_id (null), asset_type (null)
```

**Calculations:**
- **Solar capacity factor (proxy):**
  ```
  CF = (GHI / 1000) × 0.18 × temp_derating × 0.82
  temp_derating = 1 + (-0.004) × (T_c - 25)
  ```
  (Simplified PV model; 18% efficiency, temperature derating at -0.4%/°C, 18% total losses)

- **Wind power density (W/m²):**
  ```
  PD = 0.5 × 1.225 × wind_speed³
  ```
  (Using sea-level air density; 10m anemometer height)

- **Wind resource class:**
  - Poor: < 7 m/s
  - Fair: 7–8 m/s
  - Good: 8–9 m/s
  - Excellent: 9–10 m/s
  - Outstanding: ≥ 10 m/s

---

## Deployment & Execution

### Prerequisites

1. **Databricks Workspace** with Unity Catalog enabled
2. **Databricks CLI** configured (https://docs.databricks.com/dev-tools/cli/)
3. **NREL API key** (free, from https://developer.nrel.gov/signup/)

### Setup (One-Time)

1. **Create secrets scope** (if not already created):
   ```bash
   databricks secrets create-scope --scope renewable_energy
   ```

2. **Store NREL API key**:
   ```bash
   databricks secrets put --scope renewable_energy --key nrel_api_key --string-value "YOUR_NREL_API_KEY"
   ```

3. **Update workspace host** in `databricks.yml`:
   ```yaml
   targets:
     dev:
       workspace:
         host: https://your-workspace.cloud.databricks.com
   ```

### Validate Bundle

```bash
databricks bundle validate -t dev
```

### Deploy Bundle

```bash
databricks bundle deploy -t dev
```

### Run Pipeline

**Option 1: Run the full job (setup + pipeline)**
```bash
databricks bundle run renewable_generation_job -t dev
```

**Option 2: Run setup only**
```bash
databricks bundle run renewable_generation_job --task setup -t dev
```

**Option 3: Run pipeline only** (requires setup to have run first)
```bash
databricks bundle run renewable_generation_job --task refresh_medallion -t dev
```

### Monitor Execution

After running, check job/pipeline status in the Databricks UI or via CLI:
```bash
databricks jobs list --name renewable_generation_job
databricks runs list --job-name renewable_generation_job
```

---

## Configuration & Variables

### Bundle Variables (`databricks.yml`)

| Variable | Default | Description |
|----------|---------|-------------|
| `catalog` | `geo` | Unity Catalog name |
| `schema` | `renewable_energy` | Schema name within catalog |
| `start_date` | `20230101` | Data ingestion start date (YYYYMMDD) |
| `end_date` | `20231231` | Data ingestion end date (YYYYMMDD) |
| `nrel_secret_scope` | `renewable_energy` | Databricks secrets scope for API keys |
| `nrel_secret_key` | `nrel_api_key` | Secret key name for NREL API key |

**Override at deploy time:**
```bash
databricks bundle deploy -t dev -var start_date=20240101 -var end_date=20240331
databricks bundle run renewable_generation_job -t dev -var start_date=20240101 -var end_date=20240331
```

---

## Data Source Coverage & Documentation

### Solar Data

#### NASA POWER — Global, No Auth Required
- **Meteorology + solar irradiance** (GHI, DNI, DHI) for renewable energy use
- **~0.5° × 0.625°** grid (MERRA-2 / satellite based), daily & hourly
- **Public API**, no authentication required
- **Docs:** https://power.larc.nasa.gov/docs/
- **Rate limit:** ~200 requests/day per IP; use batching for large grids

#### NREL NSRDB (Himawari) — Asia-Pacific, Requires API Key
- **Himawari-8/9 satellite-based** solar irradiance
- **Covers all of Australia** at ~2 km resolution, 10-min/hourly, 2011 onward
- **Requires NREL Developer API key** (free account at https://developer.nrel.gov/signup/)
- **Docs:** https://developer.nrel.gov/docs/solar/nsrdb/

### Wind Data

#### NASA POWER (Secondary)
- Wind speed at 10m (from MERRA-2)
- Limited accuracy for wind resource assessment

#### Global Wind Atlas (Alternative)
- **Gridded mean wind speed & power density** at multiple hub heights
- Global coverage including Australia
- https://globalwindatlas.info/

#### ERA5 (Recommended for Production)
- **100m wind components** (u100, v100) from ECMWF reanalysis
- Suitable for historical wind time series and hub-height assessment
- See folder `02-gridded-data-sources/02-era5-wind/` for integration

#### AEMO (Label/Target Data)
- **Wind/solar generation actuals** (5-min, NEM + regions)
- For model validation and forecasting targets

---

## Troubleshooting

### NASA POWER API Returns 429 (Rate Limit)
- Reduce concurrent requests
- Batch by location or date range
- Retry with exponential backoff
- Contact NASA POWER support if persistent

### NREL API Key Not Found
- Verify secrets scope exists: `databricks secrets list-scopes`
- Check key is stored: `databricks secrets list --scope renewable_energy`
- Re-run: `dbutils.secrets.put(scope='renewable_energy', key='nrel_api_key', value='YOUR_KEY')`

### Silver Table Has NULLs for NSRDB
- Check NSRDB CSV column names (may differ: `GHI` vs `ghi`, `Wind Speed` vs `wind_speed`)
- Print sample schema: `spark.read.table("geo.renewable_energy.bronze_nsrdb_himawari").printSchema()`
- Update column mappings in `02_silver.sql` as needed

### Gold Aggregates Are 0 or NaN
- Verify Bronze data contains valid numeric values
- Check quality filters in `02_silver.sql` — may be too strict
- Inspect sample records: `SELECT * FROM geo.renewable_energy.silver_solar_irradiance LIMIT 100;`

### DLT Pipeline Fails to Start
- Verify all dependencies installed: `requests`, `xarray`, `pandas`, `h5py`
- Check pipeline configuration in `resources/renewable_generation.pipeline.yml`
- View pipeline run logs in Databricks UI

---

## Next Steps

1. **Expand grid coverage:** Parameterize latitude/longitude ranges in `01_bronze.py` for full Australian bounding box (lat -44…-10, lon 112…154)

2. **Integrate ERA5 winds:** Create a companion pipeline in `02-era5-wind/` for 100m hub-height wind data

3. **Add renewable asset inventory:**
   - Ingest AREMI or other solar/wind farm databases
   - Create `renewable_assets.solar_farms` and `renewable_assets.wind_farms` tables
   - Join in Gold layer via Mosaic H3 spatial indexing

4. **Enable Mosaic H3 indexing:** Uncomment and configure H3 cell ID computation in `03_gold.sql` for spatial aggregation

5. **Integrate AEMO actuals:** Ingest NEM generation data for model validation and forecasting targets

6. **Create Lakeview dashboards:** Build interactive dashboards over `gold_solar_wind_capacity_factor` for resource mapping

---

## Migration Notes (From Standalone Notebooks)

The original notebook-based pipeline has been refactored into a DAB + DLT structure:

| Original Notebook | New Structure | Type |
|-------------------|---------------|------|
| `00_setup_and_verify.py` | `src/setup/00_setup.py` | Setup task in job |
| `01_bronze_nasa_power.py` | `src/pipelines/01_bronze.py` + `bronze_nasa_power_solar` MV | DLT table (@dp.table) |
| `02_bronze_nsrdb_himawari.py` | `src/pipelines/01_bronze.py` + `bronze_nsrdb_himawari` MV | DLT table (@dp.table) |
| `03_silver_solar.py` | `src/pipelines/02_silver.sql` + `silver_solar_irradiance` MV | Materialized view (SQL) |
| `04_gold_solar_wind_capacity_factor.py` | `src/pipelines/03_gold.sql` + `gold_solar_wind_capacity_factor` MV | Materialized view (SQL) |

**Key changes:**
- ✅ All business logic preserved (API endpoints, bounding box, variables, quality filters, calculations)
- ✅ Configuration externalized to `databricks.yml` variables
- ✅ Secrets managed via Databricks secrets scope (set at setup time)
- ✅ Bronze layer uses Python @dp.table (API fetching requires imperative logic)
- ✅ Silver/Gold layers use SQL CREATE OR REFRESH MATERIALIZED VIEW (declarative, optimized)
- ✅ Materialized views automatically tracked and updated by DLT
- ✅ Partitioning by year/month/day for efficiency
- ✅ Scheduled via DAB job (configurable cron schedule)

---

## Requested Sources & Deviations

**Original request:** NSRDB, WIND Toolkit, NASA POWER (adjusted for Australia)

**Adjustments made:**
1. **NASA POWER**: ✅ Used as-is (global coverage)
2. **NREL NSRDB**: ✅ Used Himawari product (Asia-Pacific, covers all of Australia)
3. **WIND Toolkit**: ❌ NREL WIND Toolkit is **US-only (CONUS)**; substituting ERA5 and Global Wind Atlas for Australian coverage (see Next Steps)

---

## References

- **Databricks Asset Bundles:** https://docs.databricks.com/en/dev-tools/bundles/index.html
- **Lakeflow Declarative Pipelines:** https://docs.databricks.com/en/workflows/lakeflow/index.html
- **NASA POWER API:** https://power.larc.nasa.gov/docs/
- **NREL NSRDB API:** https://developer.nrel.gov/docs/solar/nsrdb/
- **Databricks Mosaic:** https://docs.databricks.com/en/lakehouse-architecture/mosaic/overview.html
- **Databricks Unity Catalog:** https://docs.databricks.com/en/data-governance/unity-catalog/index.html
