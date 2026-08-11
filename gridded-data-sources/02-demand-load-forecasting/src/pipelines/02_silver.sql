-- Databricks notebook source

-- Silver Layer: ERA5 Cleaning & Feature Engineering
--
-- Purpose:
-- - Transform Bronze ERA5 data into clean, typed, and partitioned Silver layer
-- - Kelvin → Celsius conversion
-- - Derived features (wind speed magnitude)
-- - Quality checks and filters (physically reasonable ranges)
-- - Temporal indexing (year, month)
-- - Partitioned by year/month for efficient queries
--
-- Input: bronze_era5 (raw ERA5 data)
-- Output: silver_era5 (cleaned, partitioned, derived features)

-- COMMAND ----------

-- Read pipeline configuration
-- Note: In DAB, these are set in the pipeline configuration and accessible via spark.conf
-- For SQL, we'd typically reference these through Spark configs or pass them as parameters

CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${schema}.silver_era5
PARTITIONED BY (year, month)
COMMENT 'Cleaned, typed, and partitioned ERA5 reanalysis data'
AS
WITH cleaned_data AS (
  SELECT
    -- Temporal features
    YEAR(timestamp) AS year,
    MONTH(timestamp) AS month,

    -- Original coordinates
    timestamp,
    latitude,
    longitude,

    -- Kelvin → Celsius conversion (standard constant: 273.15)
    ROUND(t2m - 273.15, 2) AS t2m_celsius,
    ROUND(t2m_dewpoint - 273.15, 2) AS t2m_dewpoint_celsius,

    -- Solar radiation (pass through, already in W/m²)
    ROUND(ssrad, 2) AS ssrad_wm2,

    -- Wind components (pass through, already in m/s)
    ROUND(u10, 2) AS u10_ms,
    ROUND(v10, 2) AS v10_ms,
    ROUND(u100, 2) AS u100_ms,
    ROUND(v100, 2) AS v100_ms

  FROM ${catalog}.${schema}.bronze_era5

  -- Quality checks: Filter out physically unreasonable data
  WHERE
    -- Non-null coordinates and time
    timestamp IS NOT NULL
    AND latitude IS NOT NULL
    AND longitude IS NOT NULL

    -- Temperature ranges (reasonable for Earth: -60°C to 60°C)
    AND (t2m - 273.15) >= -60
    AND (t2m - 273.15) <= 60

    -- Dewpoint ranges (typically below temperature, but check physical bounds: -70°C to 50°C)
    AND (t2m_dewpoint - 273.15) >= -70
    AND (t2m_dewpoint - 273.15) <= 50

    -- Solar radiation must be non-negative (W/m²)
    AND ssrad >= 0
)

-- Derive wind speed magnitude and select final columns
SELECT
  year,
  month,
  timestamp,
  latitude,
  longitude,

  -- Converted temperatures (Celsius)
  t2m_celsius,
  t2m_dewpoint_celsius,

  -- Solar radiation
  ssrad_wm2,

  -- Wind components
  u10_ms,
  v10_ms,
  u100_ms,
  v100_ms,

  -- Derived wind speed (magnitude of wind vector: sqrt(u² + v²))
  ROUND(SQRT(u10_ms * u10_ms + v10_ms * v10_ms), 2) AS wind_speed_10m,
  ROUND(SQRT(u100_ms * u100_ms + v100_ms * v100_ms), 2) AS wind_speed_100m

FROM cleaned_data

ORDER BY timestamp, latitude, longitude;

-- COMMAND ----------

-- Verify Silver table creation and contents
SELECT
  year,
  month,
  COUNT(*) AS record_count,
  COUNT(DISTINCT latitude, longitude) AS grid_cells,
  ROUND(MIN(t2m_celsius), 2) AS min_temp_c,
  ROUND(MAX(t2m_celsius), 2) AS max_temp_c,
  ROUND(AVG(t2m_celsius), 2) AS avg_temp_c,
  ROUND(AVG(wind_speed_10m), 2) AS avg_wind_speed_10m_ms
FROM ${catalog}.${schema}.silver_era5
GROUP BY year, month
ORDER BY year, month;
