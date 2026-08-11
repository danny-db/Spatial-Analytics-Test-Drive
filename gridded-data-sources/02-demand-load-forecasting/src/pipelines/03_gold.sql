-- Databricks notebook source

-- Gold Layer: NEM Region Aggregation & Feature Engineering
--
-- Purpose:
-- - Map grid cells to NEM regions (Australian National Electricity Market)
-- - Aggregate temperatures and other variables to region level
-- - Engineer demand-relevant features: HDD/CDD, lagged temperatures, rolling statistics
-- - Prepare for joining with AEMO demand data (external source)
--
-- NEM Regions: NSW1, QLD1, VIC1, SA1, TAS1
--
-- Input: silver_era5 (cleaned ERA5 data at grid-cell level)
-- Output: gold_weather_features (region-level features, ready for demand forecasting)

-- COMMAND ----------

-- PART 1: Create temporary view for NEM region mapping
-- In production, replace this with Mosaic/Sedona point-in-polygon using official NEM boundary geometries
-- See inline comments for Mosaic integration points

CREATE TEMPORARY VIEW silver_with_region AS
SELECT
  *,
  -- Simplified NEM region assignment based on lat/lon
  -- PRODUCTION NOTE: Replace with Mosaic point_in_polygon(geometry, nem_boundary_polygon)
  CASE
    WHEN latitude > -28 THEN 'QLD1'  -- Northern Queensland
    WHEN latitude > -33 THEN
      CASE
        WHEN longitude > 150 THEN 'NSW1'  -- East coast
        ELSE 'QLD1'
      END
    WHEN latitude > -34 THEN
      CASE
        WHEN longitude < 142 THEN 'SA1'  -- West / SA border
        ELSE 'NSW1'
      END
    WHEN latitude > -37 THEN
      CASE
        WHEN longitude < 142 THEN 'SA1'  -- West / SA border
        ELSE 'VIC1'
      END
    WHEN latitude > -43 THEN 'TAS1'  -- Tasmania
    ELSE 'SA1'  -- South Australia (default for southern regions)
  END AS nem_region
FROM ${catalog}.${schema}.silver_era5;

-- COMMAND ----------

-- PART 2: Regional aggregation and feature engineering
-- Use SQL window functions for lagged and rolling statistics

CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${schema}.gold_weather_features
PARTITIONED BY (nem_region, year, month)
COMMENT 'Regional (NEM) aggregated features ready for demand forecasting'
AS
WITH regional_aggregated AS (
  -- Aggregate grid-cell data to NEM region level
  SELECT
    nem_region,
    timestamp,
    year,
    month,

    -- Regional temperature aggregates
    ROUND(AVG(t2m_celsius), 2) AS temp_mean_c,
    ROUND(MIN(t2m_celsius), 2) AS temp_min_c,
    ROUND(MAX(t2m_celsius), 2) AS temp_max_c,
    ROUND(STDDEV_POP(t2m_celsius), 2) AS temp_stddev_c,

    -- Regional dewpoint aggregates
    ROUND(AVG(t2m_dewpoint_celsius), 2) AS dewpoint_mean_c,

    -- Regional solar radiation aggregates
    ROUND(AVG(ssrad_wm2), 2) AS solar_rad_mean_wm2,
    ROUND(MAX(ssrad_wm2), 2) AS solar_rad_max_wm2,

    -- Regional wind aggregates
    ROUND(AVG(wind_speed_10m), 2) AS wind_speed_10m_mean_ms,
    ROUND(AVG(wind_speed_100m), 2) AS wind_speed_100m_mean_ms

  FROM silver_with_region
  GROUP BY nem_region, timestamp, year, month
),

-- PART 3: Engineer demand-relevant features
with_demand_features AS (
  SELECT
    nem_region,
    timestamp,
    year,
    month,

    -- Raw regional aggregates
    temp_mean_c,
    temp_min_c,
    temp_max_c,
    temp_stddev_c,
    dewpoint_mean_c,
    solar_rad_mean_wm2,
    solar_rad_max_wm2,
    wind_speed_10m_mean_ms,
    wind_speed_100m_mean_ms,

    -- Demand features: HDD/CDD based on 18°C base temperature
    -- (Standard for electricity demand modeling)
    ROUND(temp_mean_c - 18.0, 2) AS temp_deviation_from_base,

    -- HDD (Heating Degree Days): demand for heating
    -- Accumulates when temp < 18°C
    CASE
      WHEN (temp_mean_c - 18.0) < 0 THEN ROUND(-(temp_mean_c - 18.0), 2)
      ELSE 0.0
    END AS hdd_proxy,

    -- CDD (Cooling Degree Days): demand for cooling
    -- Accumulates when temp > 18°C
    CASE
      WHEN (temp_mean_c - 18.0) > 0 THEN ROUND(temp_mean_c - 18.0, 2)
      ELSE 0.0
    END AS cdd_proxy,

    -- Lagged temperature features (1, 6, 24, 48 hours back)
    -- These are computed within each NEM region, ordered by timestamp
    LAG(temp_mean_c, 1) OVER (PARTITION BY nem_region ORDER BY timestamp) AS temp_lag_1h,
    LAG(temp_mean_c, 6) OVER (PARTITION BY nem_region ORDER BY timestamp) AS temp_lag_6h,
    LAG(temp_mean_c, 24) OVER (PARTITION BY nem_region ORDER BY timestamp) AS temp_lag_24h,
    LAG(temp_mean_c, 48) OVER (PARTITION BY nem_region ORDER BY timestamp) AS temp_lag_48h,

    -- Rolling statistics over 7 days (168 hours)
    -- Average, max, min temperature over the past 7 days
    ROUND(
      AVG(temp_mean_c) OVER (
        PARTITION BY nem_region
        ORDER BY timestamp
        ROWS BETWEEN 167 PRECEDING AND CURRENT ROW  -- 7 days = 168 hours
      ),
      2
    ) AS temp_rolling_7d_mean,

    MAX(temp_mean_c) OVER (
      PARTITION BY nem_region
      ORDER BY timestamp
      ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
    ) AS temp_rolling_7d_max,

    MIN(temp_mean_c) OVER (
      PARTITION BY nem_region
      ORDER BY timestamp
      ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
    ) AS temp_rolling_7d_min

  FROM regional_aggregated
),

-- PART 4: Add temporal features
with_temporal_features AS (
  SELECT
    nem_region,
    timestamp,
    CAST(DATE(timestamp) AS DATE) AS date,
    year,
    month,
    HOUR(timestamp) AS hour_of_day,
    DAYOFWEEK(timestamp) AS day_of_week,  -- 1=Sunday, 7=Saturday
    DAYOFYEAR(timestamp) AS day_of_year,

    -- Demand features
    temp_mean_c,
    temp_min_c,
    temp_max_c,
    temp_stddev_c,
    dewpoint_mean_c,
    solar_rad_mean_wm2,
    solar_rad_max_wm2,
    wind_speed_10m_mean_ms,
    wind_speed_100m_mean_ms,

    -- HDD/CDD and temperature deviation
    temp_deviation_from_base,
    hdd_proxy,
    cdd_proxy,

    -- Lagged features
    temp_lag_1h,
    temp_lag_6h,
    temp_lag_24h,
    temp_lag_48h,

    -- Rolling statistics
    temp_rolling_7d_mean,
    temp_rolling_7d_max,
    temp_rolling_7d_min

  FROM with_demand_features
)

-- Final selection and ordering
SELECT
  nem_region,
  timestamp,
  date,
  year,
  month,
  hour_of_day,
  day_of_week,
  day_of_year,

  -- Regional aggregated raw variables
  temp_mean_c,
  temp_min_c,
  temp_max_c,
  temp_stddev_c,
  dewpoint_mean_c,
  solar_rad_mean_wm2,
  solar_rad_max_wm2,
  wind_speed_10m_mean_ms,
  wind_speed_100m_mean_ms,

  -- Demand features
  temp_deviation_from_base,
  hdd_proxy,
  cdd_proxy,

  -- Lagged features
  temp_lag_1h,
  temp_lag_6h,
  temp_lag_24h,
  temp_lag_48h,

  -- Rolling statistics
  temp_rolling_7d_mean,
  temp_rolling_7d_max,
  temp_rolling_7d_min

FROM with_temporal_features

ORDER BY nem_region, timestamp;

-- COMMAND ----------

-- Verify Gold table creation
SELECT
  nem_region,
  COUNT(*) AS record_count,
  MIN(timestamp) AS start_time,
  MAX(timestamp) AS end_time,
  ROUND(MIN(temp_mean_c), 2) AS min_temp_c,
  ROUND(MAX(temp_mean_c), 2) AS max_temp_c,
  ROUND(AVG(temp_mean_c), 2) AS avg_temp_c,
  ROUND(AVG(hdd_proxy), 4) AS avg_hdd_proxy,
  ROUND(AVG(cdd_proxy), 4) AS avg_cdd_proxy
FROM ${catalog}.${schema}.gold_weather_features
GROUP BY nem_region
ORDER BY nem_region;

-- COMMAND ----------

-- Sample Gold records from NSW1 region
SELECT
  nem_region,
  timestamp,
  date,
  hour_of_day,
  day_of_week,
  temp_mean_c,
  hdd_proxy,
  cdd_proxy,
  temp_lag_24h,
  temp_rolling_7d_mean,
  solar_rad_mean_wm2,
  wind_speed_10m_mean_ms
FROM ${catalog}.${schema}.gold_weather_features
WHERE nem_region = 'NSW1'
ORDER BY timestamp
LIMIT 10;

-- COMMAND ----------

-- STUB: Prepare for joining with AEMO NEM demand data (external ingestion)
--
-- To form a complete training dataset for demand forecasting, you would:
--
-- 1. Ingest AEMO NEM regional demand data:
--    Source: https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem
--    Columns: timestamp, nem_region, demand_mw, price_aup_mwh, etc.
--
--    CREATE TABLE ${catalog}.${schema}.silver_aemo_demand AS
--    SELECT
--      nem_region,
--      DATE_TRUNC('hour', timestamp) AS timestamp,
--      AVG(demand_mw) AS demand_mw,
--      MAX(demand_mw) AS demand_peak_mw,
--      MIN(demand_mw) AS demand_min_mw
--    FROM ... -- AEMO data source
--    GROUP BY nem_region, DATE_TRUNC('hour', timestamp)
--
-- 2. Join Gold weather features with AEMO demand:
--
--    CREATE TABLE ${catalog}.${schema}.training_demand_forecasting AS
--    SELECT
--      w.*,
--      d.demand_mw,
--      d.demand_peak_mw,
--      d.demand_min_mw
--    FROM ${catalog}.${schema}.gold_weather_features w
--    LEFT JOIN ${catalog}.${schema}.silver_aemo_demand d
--      ON w.nem_region = d.nem_region
--      AND w.timestamp = d.timestamp
--    ORDER BY w.nem_region, w.timestamp
--
-- This would create the final training dataset ready for ML model development.
