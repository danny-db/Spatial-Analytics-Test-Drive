# Databricks notebook source

"""
Setup Notebook for Demand Forecasting DAB

Purpose:
- Idempotently create the Unity Catalog and Schema for weather data
- Verify Databricks features (Unity Catalog, Serverless compute)
- Check access to external data sources (GCS, if applicable)

This notebook is run ONCE per job execution, before the medallion pipeline.
"""

# COMMAND ----------

# Import parameters from job configuration
dbutils.widgets.text("catalog", "geo", "Catalog name")
dbutils.widgets.text("schema", "weather", "Schema name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Setup: Creating catalog '{catalog}' and schema '{schema}'...")

# COMMAND ----------

# Create or verify the Unity Catalog exists
# Note: The catalog may need to be created by a workspace admin if it doesn't exist
# This command will succeed idempotently if the catalog already exists

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog} COMMENT 'Geospatial and weather data'")
    print(f"✓ Catalog '{catalog}' ready")
except Exception as e:
    print(f"Note: Catalog creation may require admin. Message: {e}")

# COMMAND ----------

# Create schema within the catalog
# This is idempotent: if the schema exists, it will do nothing

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} COMMENT 'Weather and demand forecasting data'")
print(f"✓ Schema '{catalog}.{schema}' ready")

# COMMAND ----------

# Verify Unity Catalog is enabled
try:
    catalogs = spark.sql("SHOW CATALOGS").collect()
    catalog_names = [row["catalog"] for row in catalogs]
    if catalog in catalog_names:
        print(f"✓ Verified: Unity Catalog '{catalog}' exists and is accessible")
    else:
        print(f"⚠ Warning: Catalog '{catalog}' not found in available catalogs: {catalog_names}")
except Exception as e:
    print(f"⚠ Warning: Could not verify catalogs. Message: {e}")

# COMMAND ----------

# Check that we have access to the schema
try:
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"USE SCHEMA {schema}")
    print(f"✓ Verified: Can access {catalog}.{schema}")
except Exception as e:
    print(f"✗ Error: Cannot access schema. Message: {e}")
    raise

# COMMAND ----------

# Summary
print("\n" + "="*60)
print("Setup Complete")
print("="*60)
print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Pipeline will create these materialized views:")
print(f"  - {catalog}.{schema}.bronze_era5")
print(f"  - {catalog}.{schema}.silver_era5")
print(f"  - {catalog}.{schema}.gold_weather_features")
print("="*60)
