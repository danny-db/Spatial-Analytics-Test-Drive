-- Databricks notebook source
-- Gold Layer: Solar & Wind Capacity Factor & Resource Metrics
--
-- This notebook is part of a Databricks Lakeflow Declarative Pipeline (DLT).
-- It creates a materialized view that:
-- 1. Reads cleaned silver solar irradiance data
-- 2. Calculates solar capacity factor using a simplified PV model
-- 3. Calculates wind power density and resource classification
-- 4. Aggregates to daily level by location and data source
-- 5. Includes placeholders for Mosaic H3 indexing and renewable asset joins
--
-- Solar capacity factor calculation:
--   CF_proxy = (GHI / STC_irradiance) × efficiency × temp_derating × system_losses
--   Where:
--   - GHI: Global horizontal irradiance (W/m²)
--   - STC_irradiance: 1000 W/m² (standard test conditions)
--   - efficiency: 18% (typical PV module efficiency)
--   - temp_derating: (1 + α × (T_cell - T_stc))
--     - α = -0.004 (-0.4% per °C)
--     - T_stc = 25°C (standard test conditions)
--   - system_losses: 82% (18% total losses from wiring, inverter, soiling, etc.)
--
-- Wind power density = 0.5 × air_density × wind_speed³
--   Where air_density = 1.225 kg/m³ (sea level, standard atmosphere)
--
-- Partitioning: year, month, day (for efficient querying and incremental updates)

CREATE OR REFRESH MATERIALIZED VIEW gold_solar_wind_capacity_factor AS
WITH solar_data AS (
    -- Extract solar irradiance variables and pivot to wide format
    SELECT
        longitude,
        latitude,
        timestamp,
        year,
        month,
        day,
        location_name,
        data_source,
        ingestion_timestamp,
        -- Pivot irradiance and weather variables
        MAX(CASE WHEN variable = 'ghi' THEN value ELSE NULL END) AS ghi,
        MAX(CASE WHEN variable = 'dni' THEN value ELSE NULL END) AS dni,
        MAX(CASE WHEN variable = 'dhi' THEN value ELSE NULL END) AS dhi,
        MAX(CASE WHEN variable = 'temperature' THEN value ELSE NULL END) AS temperature_k,
        MAX(CASE WHEN variable IN ('wind_speed', 'wind_speed_10m') THEN value ELSE NULL END) AS wind_speed
    FROM silver_solar_irradiance
    GROUP BY
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp
),
solar_metrics AS (
    -- Calculate capacity factor metrics
    -- Constants:
    -- - STC_IRRADIANCE: 1000 W/m² (standard test conditions)
    -- - TEMP_COEFFICIENT: -0.004 (-0.4% per °C)
    -- - SYSTEM_EFFICIENCY: 0.18 (18%)
    -- - SYSTEM_LOSSES_FACTOR: 0.82 (18% losses)
    SELECT
        longitude,
        latitude,
        timestamp,
        year,
        month,
        day,
        location_name,
        data_source,
        ingestion_timestamp,
        COALESCE(ghi, 0) AS ghi,
        COALESCE(dni, 0) AS dni,
        COALESCE(dhi, 0) AS dhi,
        COALESCE(temperature_k, 288) AS temperature_k,
        COALESCE(wind_speed, 0) AS wind_speed,
        -- Convert temperature from K to °C
        (COALESCE(temperature_k, 288) - 273.15) AS temperature_c,
        -- Temperature derating factor (relative to 25°C STC)
        -- temp_derating = 1 + (-0.004) × (T_c - 25)
        (1.0 - 0.004 * ((COALESCE(temperature_k, 288) - 273.15) - 25.0)) AS temp_derating,
        -- Simplified capacity factor proxy
        -- CF_proxy = (GHI / 1000) × 0.18 × temp_derating × 0.82
        ROUND(
            (COALESCE(ghi, 0) / 1000.0) *
            0.18 *
            (1.0 - 0.004 * ((COALESCE(temperature_k, 288) - 273.15) - 25.0)) *
            0.82,
            4
        ) AS solar_capacity_factor,
        -- Wind power density (W/m²) at anemometer height
        -- PD = 0.5 × 1.225 × wind_speed³
        ROUND(
            0.5 * 1.225 * POWER(COALESCE(wind_speed, 0), 3),
            2
        ) AS wind_power_density,
        -- Wind resource class (simplified classification)
        CASE
            WHEN COALESCE(wind_speed, 0) < 7 THEN 'Poor'
            WHEN COALESCE(wind_speed, 0) < 8 THEN 'Fair'
            WHEN COALESCE(wind_speed, 0) < 9 THEN 'Good'
            WHEN COALESCE(wind_speed, 0) < 10 THEN 'Excellent'
            ELSE 'Outstanding'
        END AS wind_resource_class
    FROM solar_data
),
daily_aggregates AS (
    -- Aggregate hourly metrics to daily level by location
    SELECT
        longitude,
        latitude,
        year,
        month,
        day,
        location_name,
        data_source,
        -- Solar metrics (daily aggregates)
        ROUND(AVG(NULLIF(ghi, 0)), 2) AS ghi_daily_avg,
        ROUND(MAX(ghi), 2) AS ghi_daily_peak,
        ROUND(AVG(temperature_c), 2) AS temperature_c_daily_avg,
        ROUND(AVG(solar_capacity_factor), 4) AS solar_cf_daily_avg,
        ROUND(MAX(solar_capacity_factor), 4) AS solar_cf_daily_peak,
        -- Wind metrics (daily aggregates)
        ROUND(AVG(wind_speed), 2) AS wind_speed_daily_avg,
        ROUND(MAX(wind_speed), 2) AS wind_speed_daily_max,
        ROUND(STDDEV(wind_speed), 2) AS wind_speed_stddev,
        ROUND(AVG(wind_power_density), 2) AS wind_power_density_avg,
        ROUND(MAX(wind_power_density), 2) AS wind_power_density_max,
        -- Metadata
        COUNT(*) AS hourly_record_count,
        MAX(ingestion_timestamp) AS last_updated
    FROM solar_metrics
    GROUP BY
        longitude, latitude, year, month, day,
        location_name, data_source
)
SELECT
    longitude,
    latitude,
    year,
    month,
    day,
    location_name,
    data_source,
    -- Solar metrics
    ghi_daily_avg,
    ghi_daily_peak,
    temperature_c_daily_avg,
    solar_cf_daily_avg,
    solar_cf_daily_peak,
    -- Wind metrics
    wind_speed_daily_avg,
    wind_speed_daily_max,
    wind_speed_stddev,
    wind_power_density_avg,
    wind_power_density_max,
    -- Metadata
    'renewable_generation_v1' AS model_version,
    hourly_record_count,
    last_updated,
    -- Placeholders for future enhancements
    -- (Commented out; uncomment when Databricks Mosaic is available in workspace)
    -- H3 cell ID for spatial aggregation (requires Databricks Mosaic library):
    -- mosaic_point_to_h3(longitude, latitude, 8) AS h3_cell_id,
    CAST(NULL AS STRING) AS h3_cell_id,
    -- Asset ID and type for renewable energy infrastructure joins
    -- (Requires a separate asset ingestion pipeline, e.g., AREMI solar/wind farms):
    -- asset_id from renewable_assets.solar_farms or renewable_assets.wind_farms
    CAST(NULL AS STRING) AS asset_id,
    CAST(NULL AS STRING) AS asset_type
FROM daily_aggregates
PARTITION BY year, month, day;

-- ===== FUTURE ENHANCEMENTS =====
--
-- 1. MOSAIC H3 INDEXING (requires Databricks Mosaic library):
--    Uncomment and update the query to include:
--    - Import Mosaic in Python or use SQL functions
--    - ALTER VIEW gold_solar_wind_capacity_factor TO ADD COLUMN h3_cell_id
--    - Compute H3 index for spatial aggregation:
--      mosaic_point_to_h3(longitude, latitude, resolution=8)
--
-- 2. ASSET JOINS (requires separate renewable asset ingestion pipeline):
--    Join to AREMI or other renewable energy asset inventories:
--    - solar_farms table (location, capacity, technology)
--    - wind_farms table (location, capacity, hub_height)
--    - CREATE TABLE renewable_assets.solar_farms AS ...
--    - CREATE TABLE renewable_assets.wind_farms AS ...
--    Then join in this view:
--      LEFT JOIN renewable_assets.solar_farms ON
--        mosaic_distance(h3_cell_id, asset_location) < threshold
--
-- 3. ERA5 WIND DATA (100m hub height):
--    Integrate ECMWF ERA5 reanalysis for realistic wind resource assessment:
--    - silver_era5_winds_100m table (u100, v100 components)
--    - Compute wind speed and direction at 100m
--    - Apply wind farm power curve model
--
-- 4. AEMO GENERATION ACTUALS (validation/forecasting targets):
--    Join to AEMO renewable generation actuals for:
--    - Model validation: compare forecast CF vs. actual
--    - Forecasting targets: train models to predict generation
--    - Requires AEMO data ingest pipeline
--
-- 5. PARTITIONING OPTIMIZATION:
--    For large-scale datasets, consider additional clustering:
--    - CLUSTER BY location_name, data_source
--    - CLUSTER BY h3_cell_id (once computed)
