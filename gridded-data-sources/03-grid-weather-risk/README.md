# Use Case 03 — Grid / Weather Risk, Australia

**Goal:** Forecast weather hazards (high wind/gusts, storms, heavy rain, extreme heat) on a 0.25° grid and correlate with network assets (transmission lines, substations) to anticipate outages, bushfire risk, and dispatch stress.

**Source:** NOAA Global Forecast System (GFS)  
**Region:** Australia (lat -44° to -10°, lon 112° to 154°)  
**Architecture:** Databricks Asset Bundle (DAB) with Lakeflow Declarative Pipeline (DLT)  
**Medallion Layers:** Bronze (raw) → Silver (clean) → Gold (asset risk overlay)  
**Spatial Engine:** Databricks Mosaic (H3 hexagons + ST_* geospatial functions)

---

## Overview

This DAB implements a **medallion architecture** for weather risk forecasting:

1. **Bronze:** Ingest raw NOAA GFS GRIB2 forecast data from AWS Open Data
2. **Silver:** Clean, normalize, and derive metrics (wind speed, temperature in °C, point geometry)
3. **Gold:** Overlay with transmission line assets via H3 spatial join; compute per-asset risk scores

### Quick Start: Deploy & Run

```bash
# Validate bundle
databricks bundle validate -t dev

# Deploy to workspace
databricks bundle deploy -t dev

# Run the job (setup + pipeline)
databricks bundle run weather_risk_job -t dev

# Or run just the pipeline (assuming setup is already done)
databricks bundle run weather_risk_pipeline -t dev
```

### Data Flow

```
S3 GRIB2
  ↓ (01_bronze.py: xarray + cfgrib)
Bronze: raw_gfs_forecast (partitioned by run_time)
  ↓ (02_silver.sql: clean + normalize)
Silver: silver_gfs_forecast (partitioned by valid_time)
  ↓ (03_gold.sql: H3 join + aggregate)
Gold: gold_asset_risk_forecast (per-asset risk scores)
```

---

## Bundle Structure

```
databricks.yml                                 # Bundle config, vars, targets
resources/
  weather_risk.job.yml                         # Job: setup + pipeline tasks
  weather_risk.pipeline.yml                    # Lakeflow pipeline config
src/
  setup/
    00_setup.py                                # Setup notebook (create schema, verify prerequisites)
  pipelines/
    01_bronze.py                               # Bronze: DLT table (GRIB2 ingestion)
    02_silver.sql                              # Silver: Materialized view (clean & normalize)
    03_gold.sql                                # Gold: Materialized views (asset risk overlay)
```

### Materialized Views

| Layer | Table Name | Source | Purpose |
|-------|-----------|--------|---------|
| **Bronze** | `bronze_gfs_forecast` | NOAA GFS GRIB2 | Raw forecast data |
| **Silver** | `silver_gfs_forecast` | Bronze (cleaned) | Normalized grid with derived metrics |
| **Gold (I)** | `gold_transmission_line_buffered` | Electricity assets | Buffered lines tessellated to H3 |
| **Gold (F)** | `gold_asset_risk_forecast` | Silver + Gold(I) | Per-asset risk scores & alerts |

---

## NOAA GFS — Global Operational Forecast

### Data Source

- **Bucket:** `s3://noaa-gfs-bdp-pds/` (AWS Open Data, public, no auth)
- **Format:** GRIB2 (gridded binary)
- **Grid:** 0.25° global resolution (~28 km at equator)
- **Update Frequency:** 4× daily (00/06/12/18 UTC)
- **Forecast Horizon:** 0–16 days (~384 hours every 3h)

### Risk Variables Ingested

| Variable | GRIB2 Name | Unit | Purpose |
|----------|-----------|------|---------|
| Surface Wind Gust | GUST | m/s | Line/pole failure, vegetation risk |
| 10m u-component Wind | 10u | m/s | Sustained wind loading |
| 10m v-component Wind | 10v | m/s | Sustained wind loading |
| Accumulated Precipitation | TP (total precip) | kg/m² ≈ mm | Flooding, access risk |
| 2m Air Temperature | TMP | K | Heat waves → demand spikes, line derating |

### Key Points

- **Forecast, not history:** GFS is forward-looking (0–16 days). For historical baseline, use ERA5 (folder 02) or GFS reforecasts.
- **Australian Filter:** Applied at read time (lat -44…-10°, lon 112…154°) to reduce data volume.
- **Cycle Time:** Each GFS cycle (00/06/12/18 UTC) provides forecasts for the next 16 days.

---

## Setup: Prerequisites & Configuration

### 1. Workspace Requirements

- **Databricks Runtime:** 13.3 LTS or later (supports Mosaic + geospatial functions)
- **Compute:** Serverless or all-purpose cluster with Spark 3.3+
- **Unity Catalog:** Enabled with existing `geo` catalog

### 2. Asset Layer Dependency

The Gold layer requires transmission line assets from the sibling electricity folder:

```
Expected table: geo.electricity.silver_transmission_line (Victoria)
Columns: line_name, geom_4326, state
```

If this table doesn't exist, deploy the electricity folder first (01-gen-electricity).

### 3. Setup Task (`00_setup.py`)

The first task in the job runs setup:

```python
databricks bundle run weather_risk_job -t dev
# Task 1: setup
#   - Creates geo.weather_risk schema
#   - Verifies geo.electricity.silver_transmission_line exists
#   - Checks Mosaic geospatial functions are available
#   - Safe to re-run (idempotent)

# Task 2: refresh_medallion
#   - Runs the Lakeflow pipeline (DLT)
#   - Materializes bronze → silver → gold layers
```

### 4. Configuration (databricks.yml)

All parameters are defined as variables and can be overridden:

```yaml
variables:
  catalog: geo                           # Unity Catalog
  schema: weather_risk                   # Schema name
  au_lat_min: -44                        # Australia bounds
  au_lat_max: -10
  au_lon_min: 112
  au_lon_max: 154
  gfs_s3_bucket: s3://noaa-gfs-bdp-pds/ # AWS Open Data
  gfs_latest_cycle_date: "20240115"      # Set by orchestration
  gfs_latest_cycle_hour: "00"
  # Risk thresholds (configurable)
  gust_threshold_ms: 20
  wind_speed_threshold_ms: 15
  precip_threshold_mm: 10
  temp_threshold_c: 35
  h3_resolution: 9                       # ~172 km² per H3 cell
  line_buffer_degrees: 0.0045            # ~500m buffer
```

**To override at deploy time:**

```bash
databricks bundle deploy -t dev \
  --var="gfs_latest_cycle_date=20240116" \
  --var="gfs_latest_cycle_hour=12"
```

---

## Table Reference

### Bronze: `geo.weather_risk.bronze_gfs_forecast`

Raw NOAA GFS forecast data, one row per grid cell × forecast hour.

| Column | Type | Unit | Notes |
|--------|------|------|-------|
| lat | DOUBLE | degrees | -44 to -10 (Australia) |
| lon | DOUBLE | degrees | 112 to 154 (Australia) |
| valid_time | TIMESTAMP | UTC | Forecast valid time |
| run_time | TIMESTAMP | UTC | Forecast cycle (00/06/12/18) |
| gust | DOUBLE | m/s | Surface wind gust |
| u10m | DOUBLE | m/s | 10m u-component wind |
| v10m | DOUBLE | m/s | 10m v-component wind |
| tp | DOUBLE | kg/m² ≈ mm | Total accumulated precipitation |
| t2m | DOUBLE | K | 2m air temperature |

**Partitioned by:** `run_time`  
**Source:** DLT @dp.table (01_bronze.py)  
**Retention:** Recommend 90 days

### Silver: `geo.weather_risk.silver_gfs_forecast`

Cleaned, typed, and quality-filtered forecast grid with derived metrics and point geometry.

| Column | Type | Unit | Notes |
|--------|------|------|-------|
| lat | DOUBLE | degrees | — |
| lon | DOUBLE | degrees | — |
| valid_time | TIMESTAMP | UTC | Forecast valid time |
| run_time | TIMESTAMP | UTC | Forecast cycle |
| gust_ms | DOUBLE | m/s | Surface wind gust (≥0) |
| u10m | DOUBLE | m/s | 10m u-component wind |
| v10m | DOUBLE | m/s | 10m v-component wind |
| wind_speed_ms | DOUBLE | m/s | √(u10m² + v10m²) sustained wind |
| precip_mm | DOUBLE | mm | Accumulated precipitation |
| temp_k | DOUBLE | K | 2m air temperature (200–330K) |
| temp_c | DOUBLE | °C | 2m air temperature (derived) |
| grid_point_geom | GEOMETRY | WKT | ST_POINT(lon, lat) |

**Partitioned by:** `valid_time`  
**Source:** Materialized view SQL (02_silver.sql)

### Gold (Intermediate): `geo.weather_risk.gold_transmission_line_buffered`

Transmission line assets buffered and tessellated to H3 hexagons for efficient spatial join.

| Column | Type | Unit | Notes |
|--------|------|------|-------|
| line_name | STRING | — | Transmission line identifier |
| buffered_geom | GEOMETRY | WKT | Line buffered by 500m (0.0045°) |
| h3_cell | STRING | — | H3 hexagon at resolution 9 |

**Source:** `geo.electricity.silver_transmission_line` (Victoria)

### Gold (Final): `geo.weather_risk.gold_asset_risk_forecast`

Per-asset weather-risk scores, one row per transmission line × forecast valid time.

| Column | Type | Unit | Notes |
|--------|------|------|-------|
| line_name | STRING | — | Transmission line identifier |
| valid_time | TIMESTAMP | UTC | Forecast valid time |
| run_time | TIMESTAMP | UTC | Forecast cycle |
| max_gust_ms | DOUBLE | m/s | Max gust in asset buffer |
| max_wind_speed_ms | DOUBLE | m/s | Max sustained wind |
| max_precip_mm | DOUBLE | mm/6h | Max precipitation |
| max_temp_c | DOUBLE | °C | Max temperature |
| num_cells | INT | — | Count of forecast cells intersecting asset |
| avg_gust_ms | DOUBLE | m/s | Average gust in buffer |
| avg_wind_speed_ms | DOUBLE | m/s | Average sustained wind |
| avg_precip_mm | DOUBLE | mm/6h | Average precipitation |
| min_gust_ms | DOUBLE | m/s | Min gust (for optional analysis) |
| min_wind_speed_ms | DOUBLE | m/s | Min wind speed |
| min_precip_mm | DOUBLE | mm/6h | Min precipitation |
| min_temp_c | DOUBLE | °C | Min temperature |
| **gust_alert** | **INT** | **0/1** | **1 if max_gust_ms > 20 m/s** |
| **wind_alert** | **INT** | **0/1** | **1 if max_wind_speed_ms > 15 m/s** |
| **precip_alert** | **INT** | **0/1** | **1 if max_precip_mm > 10 mm** |
| **heat_alert** | **INT** | **0/1** | **1 if max_temp_c > 35 °C** |
| **combined_risk_score** | **INT** | **0–4** | **Sum of alerts; use for prioritization** |

**Partitioned by:** `valid_time`  
**Source:** Materialized view SQL (03_gold.sql)  

---

## Risk Alert Configuration

The Gold layer computes binary alerts and a combined risk score based on these thresholds:

| Hazard | Metric | Threshold | Rationale |
|--------|--------|-----------|-----------|
| **Gust** | max_gust_ms | > 20 m/s | High wind can damage poles/lines |
| **Sustained Wind** | max_wind_speed_ms | > 15 m/s | Sustained wind loading on conductors |
| **Precipitation** | max_precip_mm | > 10 mm/6h | Heavy rain → flooding, access risk |
| **Heat** | max_temp_c | > 35 °C | Heatwaves → line derating, demand surge |

**Combined Risk Score:** Sum of active alerts (0–4)
- 0 = No hazards detected
- 1 = One hazard type
- 2 = Two hazard types (elevated risk)
- 3 = Three hazard types (high risk)
- 4 = All four hazards (extreme risk)

**To adjust thresholds:** Edit `databricks.yml` variables or pass at deploy time.

---

## Spatial Join Architecture

### H3 Indexing for Efficiency

The Gold layer uses a **two-stage H3-indexed spatial join** to efficiently correlate forecast grid cells with transmission line assets:

1. **Index Forecast Grid:** Map each forecast cell to H3 hexagon (resolution 9)
2. **Index Assets:** Buffer transmission lines and tessellate to H3 cells
3. **Join on H3 Equality:** Fast indexed join
4. **Verify with ST_INTERSECTS:** Confirm true geometric overlap

### Performance Notes

- **H3 Resolution 9:** ~172 km² per hexagon (~300 cells per transmission line buffer)
- **Alternative:** Resolution 10 (~19 km²) for finer granularity (more cells, slower join)
- **Partitioning:** valid_time ensures efficient filtering by forecast window

---

## Common Queries

### High-Risk Assets (Combined Risk Score ≥ 2)

```sql
SELECT
  line_name,
  valid_time,
  max_gust_ms,
  max_wind_speed_ms,
  max_precip_mm,
  max_temp_c,
  combined_risk_score
FROM
  geo.weather_risk.gold_asset_risk_forecast
WHERE
  combined_risk_score >= 2
ORDER BY
  valid_time DESC, combined_risk_score DESC;
```

### Asset Risk Summary

```sql
SELECT
  line_name,
  COUNT(*) as total_forecasts,
  SUM(gust_alert) as gust_alerts,
  SUM(wind_alert) as wind_alerts,
  SUM(precip_alert) as precip_alerts,
  SUM(heat_alert) as heat_alerts,
  ROUND(AVG(combined_risk_score), 2) as avg_risk
FROM
  geo.weather_risk.gold_asset_risk_forecast
GROUP BY
  line_name
ORDER BY
  avg_risk DESC;
```

### Forecast Comparison (Latest 2 Cycles)

```sql
WITH latest_cycles AS (
  SELECT DISTINCT run_time
  FROM geo.weather_risk.gold_asset_risk_forecast
  ORDER BY run_time DESC
  LIMIT 2
)
SELECT
  run_time,
  line_name,
  valid_time,
  combined_risk_score
FROM
  geo.weather_risk.gold_asset_risk_forecast
WHERE
  run_time IN (SELECT run_time FROM latest_cycles)
ORDER BY
  run_time DESC, valid_time DESC, combined_risk_score DESC;
```

---

## Dependencies

### Databricks Runtime & Compute

- **Runtime:** DBR 13.3 LTS or later
- **Spark:** 3.3+ (required for Mosaic geospatial functions)
- **Mosaic:** ≥0.3.0 (bundled in DBR)
- **Compute:** Serverless or all-purpose cluster

### Python Packages (in Pipeline)

Installed automatically by pipeline dependencies in `weather_risk.pipeline.yml`:

- **xarray** ≥0.20 — Multi-dimensional array I/O
- **cfgrib** ≥0.9.9 — GRIB2 reader (eccodes backend)
- **s3fs** ≥2023.1 — AWS S3 file system access
- **rasterio** ≥1.3 — Raster I/O (optional)
- **netCDF4** ≥1.6 — NetCDF format support
- **pandas** ≥1.5 — DataFrames
- **numpy** ≥1.23 — Numerics

### Data Dependencies

- **NOAA GFS:** `s3://noaa-gfs-bdp-pds/` (AWS Open Data, public, no auth)
- **Electricity Assets:** `geo.electricity.silver_transmission_line` (from sibling folder)
- **Geoscience Australia (stub):** To be integrated in future iterations

---

## Known Limitations & Future Stubs

### 1. Asset Data Source

**Status:** Stub using Victoria transmission lines  
**TODO:** Replace with Geoscience Australia comprehensive inventory

- Source: Geoscience Australia Digital Atlas
- Format: GeoJSON or Shapefile
- Table: `geo.weather_risk.bronze_ga_transmission_lines`
- Coverage: All Australian transmission lines + feeders

### 2. Historical Outage Correlation

**Status:** SQL template provided (03_gold.sql)  
**TODO:** Ingest utility's internal outage records

- Table: `geo.weather_risk.historical_outages`
- Join: Gold risk forecast with outage records by asset + time window
- Model: Logistic regression or tree-based to predict outage probability

### 3. Forecast Cycle Discovery & Orchestration

**Status:** Cycle date/hour hard-coded in pipeline parameters  
**TODO:** Auto-discover latest GFS cycles from S3

- Pattern: `s3://noaa-gfs-bdp-pds/gfs.{YYYYMMDD}/{HH}/atmos/gfs.t{HH}z.pgrb2.0p25.f{FFF}`
- Recommendation: Databricks Jobs to poll S3 on schedule (4× daily) and trigger pipeline

### 4. Spatial Resolution Trade-Off

**Current:** H3 resolution 9 (~172 km²/cell)  
**Note:** May be coarse for sub-transmission feeders; resolution 10 (~19 km²) available in future

---

## Production Deployment Checklist

- [ ] Test ingestion against live AWS GFS bucket (sample 1–2 cycles)
- [ ] Validate output row counts and null distribution
- [ ] Correlate with BOM official warnings to assess forecast relevance
- [ ] Integrate asset layer from Geoscience Australia
- [ ] Ingest 6–12 months of historical outages for model training
- [ ] Build & validate outage probability model
- [ ] Set thresholds based on utility risk appetite (from current defaults)
- [ ] Schedule Bronze ingestion job (nightly for latest GFS cycles)
- [ ] Set up alerting pipeline (query Gold layer for combined_risk_score > 2)
- [ ] Document runbook for operational team
- [ ] Integrate with Databricks SQL Warehouse dashboards
- [ ] Configure alerting (e.g., email, Slack) for high-risk conditions

---

## References

- **NOAA GFS:** https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast-system
- **AWS Open Data Registry:** https://registry.opendata.aws/noaa-gfs-bdp-pds/
- **Databricks Mosaic (Geospatial SQL):** https://docs.databricks.com/en/sql/language-manual/functions/st_point.html
- **H3 Hexagonal Indexing:** https://h3geo.org/
- **xarray + cfgrib:** https://xarray-intake.readthedocs.io/en/latest/data-catalogs.html
- **Geoscience Australia:** https://www.ga.gov.au/
- **Bureau of Meteorology (BOM):** http://www.bom.gov.au/
