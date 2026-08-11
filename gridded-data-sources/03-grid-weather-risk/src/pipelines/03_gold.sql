-- Gold Layer — Network Asset Weather Risk Overlay
--
-- Two-stage medallion architecture for asset risk scoring:
--
-- Stage 1: gold_transmission_line_buffered
--   - Buffer transmission lines from electricity asset layer
--   - Tessellate buffers to H3 hexagons (resolution 9)
--   - Enables efficient spatial join with forecast grid (also in H3)
--
-- Stage 2: gold_asset_risk_forecast
--   - H3-indexed join between forecast grid and buffered lines
--   - Aggregate risk metrics per (asset, valid_time)
--   - Apply alert thresholds and compute combined risk score
--
-- Dependencies:
--   - silver_gfs_forecast (from 02_silver.sql)
--   - geo.electricity.silver_transmission_line (asset layer)
--
-- Materialized Views:
--   - gold_transmission_line_buffered (intermediate, indexed by H3)
--   - gold_asset_risk_forecast (final, one row per asset × valid_time)

-- ============================================================================
-- Stage 1: Prepare Asset Layer (Transmission Lines)
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW
  IF NOT EXISTS {{ var.catalog }}.{{ var.schema }}.gold_transmission_line_buffered
COMMENT 'Transmission line buffers tessellated to H3 hexagons for efficient spatial join'
AS
WITH line_buffer AS (
  SELECT
    line_name,
    -- Buffer each transmission line by 500m (≈ 0.0045° at equator)
    -- This captures grid cells near (but not necessarily touching) the line
    ST_BUFFER(
      ST_GEOMFROMTEXT(geom_4326),
      {{ var.line_buffer_degrees }}
    ) as buffered_geom
  FROM
    {{ var.catalog }}.electricity.silver_transmission_line
  WHERE
    -- Current stub: Victoria transmission lines only
    -- TODO: Integrate Geoscience Australia comprehensive asset inventory
    state = 'Victoria'
    AND geom_4326 IS NOT NULL
),
h3_tessellation AS (
  SELECT
    line_name,
    buffered_geom,
    -- Tessellate buffered geometry to H3 hexagons at resolution 9 (≈ 172 km²)
    -- This creates multiple H3 cells per line, enabling efficient indexed join
    -- Note: Higher resolution (e.g., 10) provides finer granularity but more cells
    h3_cell
  FROM
    line_buffer
    LATERAL VIEW EXPLODE(
      H3_COVERASH3(buffered_geom, {{ var.h3_resolution }})
    ) AS h3_cell
)
SELECT
  line_name,
  buffered_geom,
  h3_cell
FROM
  h3_tessellation
ORDER BY
  line_name, h3_cell;

-- ============================================================================
-- Stage 2: Overlay Forecast Grid with Asset Buffers (H3-Indexed Spatial Join)
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW
  IF NOT EXISTS {{ var.catalog }}.{{ var.schema }}.gold_asset_risk_forecast
PARTITIONED BY (valid_time)
COMMENT 'Per-asset weather-risk scores: one row per transmission line × forecast valid time'
AS
WITH forecast_h3 AS (
  -- Map each forecast grid cell to H3 hexagon (resolution 9)
  -- This enables efficient indexed join with transmission line H3 cells
  SELECT
    lat,
    lon,
    valid_time,
    run_time,
    gust_ms,
    u10m,
    v10m,
    wind_speed_ms,
    precip_mm,
    temp_k,
    temp_c,
    grid_point_geom,
    H3_POINTASH3(grid_point_geom, {{ var.h3_resolution }}) as h3_cell
  FROM
    {{ var.catalog }}.{{ var.schema }}.silver_gfs_forecast
),
line_forecast_join AS (
  -- H3-indexed join: forecast cells intersecting buffered transmission lines
  -- H3 equality accelerates the join; ST_INTERSECTS verifies true overlap
  SELECT
    l.line_name,
    f.valid_time,
    f.run_time,
    f.lat,
    f.lon,
    f.gust_ms,
    f.wind_speed_ms,
    f.precip_mm,
    f.temp_c,
    f.grid_point_geom,
    ST_INTERSECTS(f.grid_point_geom, l.buffered_geom) as is_intersecting
  FROM
    forecast_h3 f
  INNER JOIN
    {{ var.catalog }}.{{ var.schema }}.gold_transmission_line_buffered l
  ON
    -- H3-indexed join for efficiency
    f.h3_cell = l.h3_cell
),
risk_aggregation AS (
  -- Aggregate risk metrics per (asset, valid_time)
  -- Filter to true intersections (ST_INTERSECTS = TRUE)
  SELECT
    line_name,
    valid_time,
    run_time,
    -- Max metrics (extreme weather)
    MAX(gust_ms) as max_gust_ms,
    MAX(wind_speed_ms) as max_wind_speed_ms,
    MAX(precip_mm) as max_precip_mm,
    MAX(temp_c) as max_temp_c,
    -- Count of forecast cells intersecting asset buffer
    COUNT(*) as num_cells,
    -- Avg metrics (overall conditions)
    AVG(gust_ms) as avg_gust_ms,
    AVG(wind_speed_ms) as avg_wind_speed_ms,
    AVG(precip_mm) as avg_precip_mm,
    -- Min for optional analysis
    MIN(gust_ms) as min_gust_ms,
    MIN(wind_speed_ms) as min_wind_speed_ms,
    MIN(precip_mm) as min_precip_mm,
    MIN(temp_c) as min_temp_c
  FROM
    line_forecast_join
  WHERE
    -- Only include true geometric intersections
    is_intersecting = TRUE
  GROUP BY
    line_name, valid_time, run_time
)
SELECT
  line_name,
  valid_time,
  run_time,
  -- Risk metrics (max, avg, min)
  max_gust_ms,
  max_wind_speed_ms,
  max_precip_mm,
  max_temp_c,
  num_cells,
  avg_gust_ms,
  avg_wind_speed_ms,
  avg_precip_mm,
  min_gust_ms,
  min_wind_speed_ms,
  min_precip_mm,
  min_temp_c,
  -- Risk alerts (binary): 1 if threshold exceeded, 0 otherwise
  -- Thresholds are configurable via pipeline parameters
  CASE
    WHEN max_gust_ms > {{ var.gust_threshold_ms }} THEN 1
    ELSE 0
  END as gust_alert,
  CASE
    WHEN max_wind_speed_ms > {{ var.wind_speed_threshold_ms }} THEN 1
    ELSE 0
  END as wind_alert,
  CASE
    WHEN max_precip_mm > {{ var.precip_threshold_mm }} THEN 1
    ELSE 0
  END as precip_alert,
  CASE
    WHEN max_temp_c > {{ var.temp_threshold_c }} THEN 1
    ELSE 0
  END as heat_alert,
  -- Combined risk score: sum of alerts (0-4)
  -- Use for prioritization: high score (≥2) indicates multiple hazards
  (
    CASE WHEN max_gust_ms > {{ var.gust_threshold_ms }} THEN 1 ELSE 0 END +
    CASE WHEN max_wind_speed_ms > {{ var.wind_speed_threshold_ms }} THEN 1 ELSE 0 END +
    CASE WHEN max_precip_mm > {{ var.precip_threshold_mm }} THEN 1 ELSE 0 END +
    CASE WHEN max_temp_c > {{ var.temp_threshold_c }} THEN 1 ELSE 0 END
  ) as combined_risk_score
FROM
  risk_aggregation
ORDER BY
  valid_time DESC, line_name, combined_risk_score DESC;

-- ============================================================================
-- Data Quality & Validation Queries
-- ============================================================================

-- Summary: Asset-level risk distribution
SELECT
  line_name,
  COUNT(*) as total_forecasts,
  SUM(gust_alert) as gust_alert_count,
  SUM(wind_alert) as wind_alert_count,
  SUM(precip_alert) as precip_alert_count,
  SUM(heat_alert) as heat_alert_count,
  SUM(
    CASE WHEN combined_risk_score >= 2 THEN 1 ELSE 0 END
  ) as high_risk_forecasts,
  ROUND(AVG(combined_risk_score), 2) as avg_risk_score
FROM
  {{ var.catalog }}.{{ var.schema }}.gold_asset_risk_forecast
GROUP BY
  line_name
ORDER BY
  high_risk_forecasts DESC, avg_risk_score DESC;

-- High-risk periods (combined_risk_score >= 2)
SELECT
  line_name,
  valid_time,
  run_time,
  max_gust_ms,
  max_wind_speed_ms,
  max_precip_mm,
  max_temp_c,
  num_cells,
  CONCAT_WS(',',
    CASE WHEN gust_alert = 1 THEN 'GUST' END,
    CASE WHEN wind_alert = 1 THEN 'WIND' END,
    CASE WHEN precip_alert = 1 THEN 'PRECIP' END,
    CASE WHEN heat_alert = 1 THEN 'HEAT' END
  ) as active_alerts,
  combined_risk_score
FROM
  {{ var.catalog }}.{{ var.schema }}.gold_asset_risk_forecast
WHERE
  combined_risk_score >= 2
ORDER BY
  valid_time DESC, combined_risk_score DESC
LIMIT 500;

-- ============================================================================
-- STUBS: Future Integration Points
-- ============================================================================

-- TODO 1: Geoscience Australia Asset Integration
-- Current: Uses geo.electricity.silver_transmission_line (Victoria stub)
-- Future: Replace with comprehensive Geoscience Australia transmission/feeder inventory
--
-- Example schema for GA transmission lines:
-- CREATE TABLE geo.weather_risk.bronze_ga_transmission_lines (
--   asset_id STRING,
--   asset_name STRING,
--   asset_type STRING,  -- 'transmission_line', 'distribution_feeder', 'substation'
--   state STRING,
--   voltage_kv DOUBLE,
--   geometry STRING,    -- WKT (LINESTRING or POINT)
--   source STRING       -- e.g., 'Geoscience Australia'
-- )

-- TODO 2: Historical Outage Correlation
-- Correlate Gold risk forecasts with utility's internal outage records
-- to build predictive model for outage probability
--
-- Example schema for outage records:
-- CREATE TABLE geo.weather_risk.historical_outages (
--   asset_id STRING,
--   asset_name STRING,
--   outage_start TIMESTAMP,
--   outage_end TIMESTAMP,
--   duration_minutes INT,
--   outage_cause STRING,
--   weather_conditions STRING  -- free-form notes, e.g., "high wind + rain"
-- )
--
-- Example join (stub):
-- SELECT
--   g.line_name,
--   g.valid_time,
--   g.combined_risk_score,
--   o.outage_start,
--   o.duration_minutes,
--   DATEDIFF(minute, g.valid_time, o.outage_start) as minutes_before_outage
-- FROM
--   gold_asset_risk_forecast g
-- LEFT JOIN
--   historical_outages o
-- ON
--   g.line_name = o.asset_name
--   AND g.valid_time BETWEEN DATE_SUB(o.outage_start, 24) AND o.outage_start
-- WHERE
--   combined_risk_score >= 2

-- TODO 3: Forecast Cycle Discovery & Orchestration
-- Current: Cycle date/hour hard-coded in pipeline parameters
-- Future: Auto-discover latest available GFS cycles from S3
--
-- Pattern: s3://noaa-gfs-bdp-pds/gfs.{YYYYMMDD}/{HH}/atmos/...
-- Recommendation: Use Databricks Jobs to poll S3 on schedule (4x daily)
-- and trigger pipeline with latest cycle parameters
