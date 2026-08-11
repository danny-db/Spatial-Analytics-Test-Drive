# Databricks notebook source

"""
Setup notebook for Weather Risk medallion pipeline.

Idempotently:
1. Create Unity Catalog and schema
2. Verify prerequisites (asset layer, geospatial functions)
3. Create volumes if needed for external data
4. Log configuration

Safe to re-run. Does NOT install libraries (pipeline env handles dependencies).
"""

# COMMAND ----------

# Read configuration from spark.conf (passed by pipeline configuration block)
catalog = spark.conf.get("catalog", "geo")
schema = spark.conf.get("schema", "weather_risk")

au_lat_min = spark.conf.get("au_lat_min", "-44")
au_lat_max = spark.conf.get("au_lat_max", "-10")
au_lon_min = spark.conf.get("au_lon_min", "112")
au_lon_max = spark.conf.get("au_lon_max", "154")

gfs_s3_bucket = spark.conf.get("gfs_s3_bucket", "s3://noaa-gfs-bdp-pds/")
h3_resolution = spark.conf.get("h3_resolution", "9")

print(f"Configuration:")
print(f"  Catalog: {catalog}")
print(f"  Schema: {schema}")
print(f"  AU Bounds: lat [{au_lat_min}, {au_lat_max}], lon [{au_lon_min}, {au_lon_max}]")
print(f"  GFS S3: {gfs_s3_bucket}")
print(f"  H3 Resolution: {h3_resolution}")

# COMMAND ----------

# Create catalog if it doesn't exist
# Note: In UC-enabled workspaces, catalogs are typically pre-created by admins.
# This ensures the schema will be created in the correct catalog.

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    print(f"✓ Catalog '{catalog}' ready")
except Exception as e:
    print(f"⚠ Catalog creation: {e}")

# COMMAND ----------

# Create schema in the catalog
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} COMMENT 'Weather Risk medallion pipeline'")
    print(f"✓ Schema '{catalog}.{schema}' created")
except Exception as e:
    print(f"⚠ Schema creation: {e}")

# COMMAND ----------

# Verify prerequisite: Transmission line asset table exists
# This table should exist from the electricity folder (01-gen-electricity)

try:
    asset_table = f"{catalog}.electricity.silver_transmission_line"
    result = spark.sql(f"SELECT COUNT(*) as cnt FROM {asset_table} WHERE state = 'Victoria'")
    count = result.collect()[0]["cnt"]
    print(f"✓ Asset table '{asset_table}' verified ({count} Victoria transmission lines)")
except Exception as e:
    print(f"✗ Asset table verification failed: {e}")
    print(f"  Expected: {asset_table}")
    print(f"  Ensure the electricity folder (01-gen-electricity) has been ingested first.")
    raise

# COMMAND ----------

# Verify Mosaic/Geospatial functions are available
try:
    test = spark.sql("SELECT H3_POINTASH3(ST_POINT(0.0, 0.0), 9) as h3_cell")
    print(f"✓ Geospatial functions (H3, ST_*) verified")
except Exception as e:
    print(f"✗ Geospatial functions not available: {e}")
    print(f"  Ensure Databricks Runtime ≥ 13.3 LTS with Mosaic enabled")
    raise

# COMMAND ----------

# Log summary
print("\n" + "="*60)
print("Setup Complete")
print("="*60)
print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Ready for pipeline ingestion from: {gfs_s3_bucket}")
print("\nNext: Run pipeline task (refresh_medallion) to ingest GRIB2 data")
