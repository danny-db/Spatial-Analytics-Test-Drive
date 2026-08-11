-- Databricks notebook source
-- Silver Layer: Solar Irradiance Cleaning & Normalization
--
-- This notebook is part of a Databricks Lakeflow Declarative Pipeline (DLT).
-- It creates a materialized view that:
-- 1. Reads both bronze tables (NASA POWER and NSRDB Himawari)
-- 2. Normalizes schemas (handles different column names and types)
-- 3. Applies quality filters (removes nulls, out-of-range values)
-- 4. Converts to long format: one row per (lat, lon, timestamp, variable)
-- 5. Partitions by date (year, month, day) for efficient querying
--
-- Configuration (from DAB pipeline):
-- - catalog and schema are set at the pipeline level
--
-- Quality filters:
-- - Irradiance (GHI/DNI/DHI): 0 to 1500 W/m² (physical limits)
-- - Temperature: reasonable ranges (K or °C depending on source)
-- - Wind speed: >= 0
-- - Timestamps: not null

-- IMPORTANT: In DLT, we use @dp.view or @dp.table Python decorators, but since this is SQL,
-- we must wrap it. The Python wrapper file or DLT SQL syntax allows us to use:
-- CREATE OR REFRESH MATERIALIZED VIEW <name> AS SELECT ...

-- However, in a pure SQL notebook in DLT, we can't use CREATE OR REFRESH directly.
-- Instead, we'll use a two-step approach:
-- 1. Create a temporary view with the cleaned data
-- 2. Define the materialized view using dp.create_view or similar

-- For Databricks DLT SQL notebooks, we use this pattern:

CREATE OR REFRESH MATERIALIZED VIEW silver_solar_irradiance AS
WITH nasa_power_normalized AS (
    -- Normalize NASA POWER bronze data
    -- Columns: time, latitude, longitude, ALLSKY_SFC_SW_DWN, T2M, WS10M, location_name, data_source, ingestion_timestamp
    SELECT
        CAST(longitude AS DOUBLE) AS longitude,
        CAST(latitude AS DOUBLE) AS latitude,
        CAST(CAST(time AS TIMESTAMP) AS TIMESTAMP) AS timestamp,
        CAST(ALLSKY_SFC_SW_DWN AS DOUBLE) AS ghi,
        CAST(T2M AS DOUBLE) AS temperature,
        CAST(WS10M AS DOUBLE) AS wind_speed_10m,
        NULL AS dni,
        NULL AS dhi,
        location_name,
        data_source,
        ingestion_timestamp,
        YEAR(CAST(time AS TIMESTAMP)) AS year,
        MONTH(CAST(time AS TIMESTAMP)) AS month,
        DAYOFMONTH(CAST(time AS TIMESTAMP)) AS day
    FROM bronze_nasa_power_solar
    WHERE
        CAST(time AS TIMESTAMP) IS NOT NULL
        AND longitude IS NOT NULL
        AND latitude IS NOT NULL
),
nsrdb_normalized AS (
    -- Normalize NSRDB Himawari bronze data
    -- Columns: Year, Month, Day, Hour, GHI, DNI, DHI, Temperature, Wind Speed, ..., latitude, longitude, location_name, data_source, ingestion_timestamp
    -- Build timestamp from Year, Month, Day, Hour columns
    SELECT
        CAST(longitude AS DOUBLE) AS longitude,
        CAST(latitude AS DOUBLE) AS latitude,
        CAST(
            CONCAT_WS('-', Year, Month, Day, Hour),
            'yyyy-M-d-H'
        ) AS timestamp,
        CAST(GHI AS DOUBLE) AS ghi,
        CAST(DNI AS DOUBLE) AS dni,
        CAST(DHI AS DOUBLE) AS dhi,
        CAST(Temperature AS DOUBLE) AS temperature,
        CAST(`Wind Speed` AS DOUBLE) AS wind_speed_10m,
        NULL AS wind_speed_10m_nsrdb,
        location_name,
        data_source,
        ingestion_timestamp,
        CAST(Year AS INT) AS year,
        CAST(Month AS INT) AS month,
        CAST(Day AS INT) AS day
    FROM bronze_nsrdb_himawari
    WHERE
        Year IS NOT NULL
        AND Month IS NOT NULL
        AND Day IS NOT NULL
        AND Hour IS NOT NULL
),
combined_long_format AS (
    -- Union NASA POWER and NSRDB data, then melt to long format
    -- One row per (lat, lon, timestamp, variable)

    -- NASA POWER: GHI
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'ghi' AS variable, ghi AS value, 'W/m²' AS unit
    FROM nasa_power_normalized
    WHERE ghi IS NOT NULL AND NOT ISNAN(ghi)
    UNION ALL

    -- NASA POWER: Temperature (K)
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'temperature' AS variable, temperature AS value, 'K' AS unit
    FROM nasa_power_normalized
    WHERE temperature IS NOT NULL AND NOT ISNAN(temperature)
    UNION ALL

    -- NASA POWER: Wind Speed at 10m
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'wind_speed_10m' AS variable, wind_speed_10m AS value, 'm/s' AS unit
    FROM nasa_power_normalized
    WHERE wind_speed_10m IS NOT NULL AND NOT ISNAN(wind_speed_10m)
    UNION ALL

    -- NSRDB: GHI
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'ghi' AS variable, ghi AS value, 'W/m²' AS unit
    FROM nsrdb_normalized
    WHERE ghi IS NOT NULL AND NOT ISNAN(ghi)
    UNION ALL

    -- NSRDB: DNI
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'dni' AS variable, dni AS value, 'W/m²' AS unit
    FROM nsrdb_normalized
    WHERE dni IS NOT NULL AND NOT ISNAN(dni)
    UNION ALL

    -- NSRDB: DHI
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'dhi' AS variable, dhi AS value, 'W/m²' AS unit
    FROM nsrdb_normalized
    WHERE dhi IS NOT NULL AND NOT ISNAN(dhi)
    UNION ALL

    -- NSRDB: Temperature
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'temperature' AS variable, temperature AS value, 'K' AS unit
    FROM nsrdb_normalized
    WHERE temperature IS NOT NULL AND NOT ISNAN(temperature)
    UNION ALL

    -- NSRDB: Wind Speed
    SELECT
        longitude, latitude, timestamp, year, month, day,
        location_name, data_source, ingestion_timestamp,
        'wind_speed' AS variable, wind_speed_10m AS value, 'm/s' AS unit
    FROM nsrdb_normalized
    WHERE wind_speed_10m IS NOT NULL AND NOT ISNAN(wind_speed_10m)
)
SELECT
    longitude,
    latitude,
    timestamp,
    year,
    month,
    day,
    location_name,
    data_source,
    variable,
    value,
    unit,
    ingestion_timestamp
FROM combined_long_format
WHERE
    -- Quality filter: not null and not NaN
    value IS NOT NULL
    AND NOT ISNAN(value)
    AND timestamp IS NOT NULL
    -- Physical limits for irradiance (GHI/DNI/DHI)
    -- Valid range: 0 to 1500 W/m²
    AND CASE
        WHEN variable IN ('ghi', 'dni', 'dhi')
        THEN value >= 0 AND value <= 1500
        ELSE TRUE
    END
    -- Valid range for temperature (K): 250 to 330 (roughly -23°C to 57°C)
    AND CASE
        WHEN variable = 'temperature'
        THEN value >= 250 AND value <= 330
        ELSE TRUE
    END
    -- Valid range for wind speed: >= 0
    AND CASE
        WHEN variable IN ('wind_speed', 'wind_speed_10m')
        THEN value >= 0
        ELSE TRUE
    END
PARTITION BY year, month, day;
