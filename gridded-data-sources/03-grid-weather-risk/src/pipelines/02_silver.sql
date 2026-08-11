-- Silver Layer — Clean & Normalize Forecast Data
--
-- Transforms Bronze data to one row per (grid-cell, valid_time) with:
-- - Type conversions
-- - Quality filters
-- - Derived metrics (wind_speed, temp_c, point geometry)
--
-- Dependencies:
--   - bronze_gfs_forecast (from 01_bronze.py)
--
-- Materialized View: silver_gfs_forecast (partitioned by valid_time)

CREATE OR REFRESH MATERIALIZED VIEW
  IF NOT EXISTS {{ var.catalog }}.{{ var.schema }}.silver_gfs_forecast
PARTITIONED BY (valid_time)
COMMENT 'Cleaned and normalized NOAA GFS forecast grid with derived metrics'
AS
WITH raw_forecast AS (
  SELECT
    CAST(lat AS DOUBLE) as lat,
    CAST(lon AS DOUBLE) as lon,
    CAST(valid_time AS TIMESTAMP) as valid_time,
    CAST(run_time AS TIMESTAMP) as run_time,
    CAST(gust AS DOUBLE) as gust_ms,
    CAST(u10m AS DOUBLE) as u10m,
    CAST(v10m AS DOUBLE) as v10m,
    CAST(tp AS DOUBLE) as precip_mm,
    CAST(t2m AS DOUBLE) as temp_k
  FROM
    {{ var.catalog }}.{{ var.schema }}.bronze_gfs_forecast
  WHERE
    -- Quality filters
    lat IS NOT NULL
    AND lon IS NOT NULL
    AND valid_time IS NOT NULL
    AND run_time IS NOT NULL
    -- Gust must be non-negative (can be NULL for missing values)
    AND (gust_ms >= 0 OR gust_ms IS NULL)
    -- Temperature in physically plausible range (200K ≈ -73°C, 330K ≈ +57°C)
    AND (temp_k BETWEEN 200 AND 330 OR temp_k IS NULL)
)
SELECT
  lat,
  lon,
  valid_time,
  run_time,
  gust_ms,
  u10m,
  v10m,
  precip_mm,
  temp_k,
  -- Derived metric: Wind speed from u/v components (magnitude)
  SQRT(POWER(u10m, 2) + POWER(v10m, 2)) as wind_speed_ms,
  -- Derived metric: Temperature in Celsius
  temp_k - 273.15 as temp_c,
  -- Geometry: Point location in lon/lat (WGS84)
  ST_POINT(lon, lat) as grid_point_geom
FROM
  raw_forecast
ORDER BY
  valid_time, run_time, lat, lon;

-- COMMAND ----------
-- Data Quality Summary

-- Verify schema and row count
SELECT
  COUNT(*) as total_rows,
  COUNT(DISTINCT valid_time) as distinct_valid_times,
  COUNT(DISTINCT run_time) as distinct_run_times,
  MIN(valid_time) as min_valid_time,
  MAX(valid_time) as max_valid_time,
  MIN(lat) as min_lat,
  MAX(lat) as max_lat,
  MIN(lon) as min_lon,
  MAX(lon) as max_lon
FROM
  {{ var.catalog }}.{{ var.schema }}.silver_gfs_forecast;

-- Check for nulls and extreme values
SELECT
  'gust_ms' as metric,
  COUNT(*) as total,
  COUNT(CASE WHEN gust_ms IS NULL THEN 1 END) as nulls,
  ROUND(MIN(gust_ms), 2) as min_val,
  ROUND(MAX(gust_ms), 2) as max_val,
  ROUND(AVG(gust_ms), 2) as avg_val
FROM {{ var.catalog }}.{{ var.schema }}.silver_gfs_forecast
UNION ALL
SELECT
  'wind_speed_ms' as metric,
  COUNT(*) as total,
  COUNT(CASE WHEN wind_speed_ms IS NULL THEN 1 END) as nulls,
  ROUND(MIN(wind_speed_ms), 2) as min_val,
  ROUND(MAX(wind_speed_ms), 2) as max_val,
  ROUND(AVG(wind_speed_ms), 2) as avg_val
FROM {{ var.catalog }}.{{ var.schema }}.silver_gfs_forecast
UNION ALL
SELECT
  'precip_mm' as metric,
  COUNT(*) as total,
  COUNT(CASE WHEN precip_mm IS NULL THEN 1 END) as nulls,
  ROUND(MIN(precip_mm), 2) as min_val,
  ROUND(MAX(precip_mm), 2) as max_val,
  ROUND(AVG(precip_mm), 2) as avg_val
FROM {{ var.catalog }}.{{ var.schema }}.silver_gfs_forecast
UNION ALL
SELECT
  'temp_c' as metric,
  COUNT(*) as total,
  COUNT(CASE WHEN temp_c IS NULL THEN 1 END) as nulls,
  ROUND(MIN(temp_c), 2) as min_val,
  ROUND(MAX(temp_c), 2) as max_val,
  ROUND(AVG(temp_c), 2) as avg_val
FROM {{ var.catalog }}.{{ var.schema }}.silver_gfs_forecast;
