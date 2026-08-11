# Databricks notebook source

"""
Bronze Layer — NOAA GFS GRIB2 Ingestion via DLT

Reads GRIB2 forecast files from AWS Open Data bucket (s3://noaa-gfs-bdp-pds/),
filters to Australian bounding box and risk variables, and materializes as
bronze_gfs_forecast table partitioned by run_time (forecast cycle).

Note: GFS is a FORECAST (forward-looking), not a historical reanalysis.
Each file contains a single forecast cycle with multiple forecast hours.

Data Flow:
  S3 GRIB2 → xarray + cfgrib → pandas DF → Spark DF → Delta table (bronze)

Materialized View: Persisted for downstream Silver/Gold layers
"""

from pyspark import pipelines as dp
import pandas as pd
import numpy as np
from datetime import datetime
import s3fs
import xarray as xr
from pyspark.sql.types import (
    StructType, StructField, DoubleType, TimestampType, StringType
)

# COMMAND ----------

# Read configuration from pipeline context
catalog = spark.conf.get("catalog", "geo")
schema = spark.conf.get("schema", "weather_risk")

au_lat_min = float(spark.conf.get("au_lat_min", "-44"))
au_lat_max = float(spark.conf.get("au_lat_max", "-10"))
au_lon_min = float(spark.conf.get("au_lon_min", "112"))
au_lon_max = float(spark.conf.get("au_lon_max", "154"))

gfs_s3_bucket = spark.conf.get("gfs_s3_bucket", "s3://noaa-gfs-bdp-pds/")
gfs_latest_cycle_date = spark.conf.get("gfs_latest_cycle_date", "20240115")
gfs_latest_cycle_hour = spark.conf.get("gfs_latest_cycle_hour", "00")

print(f"Bronze Configuration:")
print(f"  Catalog: {catalog}, Schema: {schema}")
print(f"  AU Bounds: lat [{au_lat_min}, {au_lat_max}], lon [{au_lon_min}, {au_lon_max}]")
print(f"  GFS Cycle: {gfs_latest_cycle_date}/{gfs_latest_cycle_hour}")

# COMMAND ----------

# Helper: Read and filter GFS GRIB2 from S3

def read_gfs_grib_from_s3(s3_path: str) -> pd.DataFrame:
    """
    Read a NOAA GFS GRIB2 file from S3, filter to AU bounding box and risk variables.

    Parameters:
    -----------
    s3_path : str
        Full S3 path to GRIB2 file
        (e.g., s3://noaa-gfs-bdp-pds/gfs.20240115/00/atmos/gfs.t00z.pgrb2.0p25.f000)

    Returns:
    --------
    df : pd.DataFrame or None
        Flattened DataFrame with columns:
        - lat, lon, valid_time, run_time
        - gust, u10m, v10m, tp (total precip), t2m
        Returns None if read fails.

    Notes:
    - Uses s3fs with anonymous credentials (AWS Open Data)
    - Filters to Australian lat/lon at read time for efficiency
    - Selects only risk-relevant variables
    """
    try:
        # Open S3 file with anonymous credentials
        fs = s3fs.S3FileSystem(anon=True)

        # Read GRIB2 with xarray + cfgrib backend
        with fs.open(s3_path, "rb") as f:
            ds = xr.open_dataset(f, engine="cfgrib")

        # Extract run_time from dataset attributes (forecast cycle time)
        # Fallback to the current time if attribute is missing
        run_time_dt = None
        if "edition" in ds.attrs:
            try:
                run_time_dt = pd.to_datetime(ds.attrs["edition"])
            except:
                pass
        if run_time_dt is None:
            run_time_dt = datetime.now()

        # Filter to Australian bounding box at read time
        # Note: slice() requires max first for descending coordinate arrays
        ds = ds.sel(
            latitude=slice(au_lat_max, au_lat_min),
            longitude=slice(au_lon_min, au_lon_max),
        )

        # Select only risk variables that are present in the dataset
        risk_variables = ["gust", "u10m", "v10m", "tp", "t2m"]
        available_vars = [v for v in risk_variables if v in ds.data_vars]

        if not available_vars:
            print(f"  Warning: No risk variables found in {s3_path}")
            return None

        ds = ds[available_vars]

        # Convert to DataFrame: one row per (lat, lon, time, variable)
        # Then pivot to get variables as columns
        df = ds.to_array(dim="variable").to_pandas().reset_index()
        df.columns = ["latitude", "longitude", "time", "variable", "value"]

        # Pivot: each variable becomes a column
        df = df.pivot_table(
            index=["latitude", "longitude", "time"],
            columns="variable",
            values="value",
        ).reset_index()

        # Add run_time (forecast cycle) and rename time to valid_time
        df["run_time"] = run_time_dt
        df["valid_time"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True)

        # Rename latitude/longitude to lat/lon
        df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

        # Reorder columns for readability
        col_order = ["lat", "lon", "valid_time", "run_time"] + [
            c for c in df.columns
            if c not in ["lat", "lon", "valid_time", "run_time"]
        ]
        df = df[col_order]

        return df

    except Exception as e:
        print(f"  Error reading {s3_path}: {e}")
        return None


# COMMAND ----------

# Helper: Ingest a complete GFS forecast cycle


def ingest_gfs_forecast_cycle(cycle_date: str, cycle_hour: int) -> pd.DataFrame:
    """
    Ingest all forecast hours from a single GFS forecast cycle.

    Parameters:
    -----------
    cycle_date : str
        Format: 'YYYYMMDD' (e.g., '20240115')
    cycle_hour : int
        0, 6, 12, or 18 UTC

    Returns:
    --------
    df_cycle : pd.DataFrame or None
        Union of all forecast hours for this cycle, or None if no data.

    Notes:
    - GFS provides forecast hours: f000, f003, f006, ..., f384 (every 3h)
    - f000 = analysis (nowcast), f003+ = forecast
    - We ingest all available forecast hours
    """
    dfs = []

    # Forecast hours to ingest: 0 to 384 in 3-hour increments (16 days)
    # Note: GFS also provides shorter-range high-res forecasts (f000-f120 every 1h),
    # but we use the standard 3-hourly resolution for all hours here.
    forecast_hours = list(range(0, 385, 3))

    print(f"Ingesting GFS cycle {cycle_date}/{cycle_hour:02d} ({len(forecast_hours)} forecast hours)...")

    for fh in forecast_hours:
        s3_path = (
            f"{gfs_s3_bucket}gfs.{cycle_date}/"
            f"{cycle_hour:02d}/atmos/gfs.t{cycle_hour:02d}z.pgrb2.0p25.f{fh:03d}"
        )

        df = read_gfs_grib_from_s3(s3_path)
        if df is not None:
            dfs.append(df)
            print(f"  ✓ f{fh:03d} ({len(df)} rows)")
        else:
            print(f"  ✗ f{fh:03d} (read failed or no data)")

    if dfs:
        df_cycle = pd.concat(dfs, ignore_index=True)
        print(f"Total: {len(df_cycle)} rows for cycle {cycle_date}/{cycle_hour:02d}")
        return df_cycle
    else:
        print(f"No data ingested for cycle {cycle_date}/{cycle_hour:02d}")
        return None


# COMMAND ----------

# Define Bronze layer table as DLT materialized view


@dp.table(
    name="bronze_gfs_forecast",
    description="Raw NOAA GFS forecast data, one row per grid cell × forecast hour",
    comment="Partitioned by run_time (forecast cycle: 00/06/12/18 UTC)",
)
@dp.expect_or_drop("valid_lat_lon", "(lat >= -90 AND lat <= 90) AND (lon >= -180 AND lon <= 180)")
@dp.expect_or_drop("valid_gust", "gust >= 0 OR gust IS NULL")
@dp.expect_or_drop("valid_temp_k", "(t2m >= 200 AND t2m <= 330) OR t2m IS NULL")
def bronze_gfs_forecast():
    """
    Bronze layer: Raw NOAA GFS forecast data from AWS Open Data.

    Schema:
    -------
    - lat (DOUBLE): Latitude (-44 to -10 for Australia filter)
    - lon (DOUBLE): Longitude (112 to 154 for Australia filter)
    - valid_time (TIMESTAMP): Forecast valid time (UTC)
    - run_time (TIMESTAMP): Forecast cycle time (00/06/12/18 UTC)
    - gust (DOUBLE): Surface wind gust (m/s)
    - u10m (DOUBLE): 10m u-component wind (m/s)
    - v10m (DOUBLE): 10m v-component wind (m/s)
    - tp (DOUBLE): Total accumulated precipitation (kg/m² → effectively mm)
    - t2m (DOUBLE): 2m air temperature (K)

    Data Quality:
    - Filtered to Australian bounding box at read time
    - Quality checks via DLT expects (lat/lon bounds, gust ≥ 0, temp 200-330K)
    - Partitioned by run_time for efficient queries by forecast cycle

    Retention:
    - Recommend: 90 days (3-month rolling window of forecasts)
    """

    # Ingest the configured GFS cycle
    df_cycle = ingest_gfs_forecast_cycle(gfs_latest_cycle_date, int(gfs_latest_cycle_hour))

    if df_cycle is None or len(df_cycle) == 0:
        # Return empty DataFrame with correct schema if no data
        schema = StructType(
            [
                StructField("lat", DoubleType()),
                StructField("lon", DoubleType()),
                StructField("valid_time", TimestampType()),
                StructField("run_time", TimestampType()),
                StructField("gust", DoubleType()),
                StructField("u10m", DoubleType()),
                StructField("v10m", DoubleType()),
                StructField("tp", DoubleType()),
                StructField("t2m", DoubleType()),
            ]
        )
        return spark.createDataFrame([], schema)

    # Convert pandas DataFrame to Spark DataFrame
    sdf = spark.createDataFrame(df_cycle)

    # Ensure proper column types
    sdf = sdf.withColumn("lat", sdf["lat"].cast(DoubleType())) \
             .withColumn("lon", sdf["lon"].cast(DoubleType())) \
             .withColumn("valid_time", sdf["valid_time"].cast(TimestampType())) \
             .withColumn("run_time", sdf["run_time"].cast(TimestampType())) \
             .withColumn("gust", sdf["gust"].cast(DoubleType())) \
             .withColumn("u10m", sdf["u10m"].cast(DoubleType())) \
             .withColumn("v10m", sdf["v10m"].cast(DoubleType())) \
             .withColumn("tp", sdf["tp"].cast(DoubleType())) \
             .withColumn("t2m", sdf["t2m"].cast(DoubleType()))

    return sdf

# COMMAND ----------

print("✓ Bronze layer (01_bronze.py) loaded")
print("  DLT table: bronze_gfs_forecast")
print("  Dependencies: xarray, cfgrib, s3fs, pandas, numpy")
