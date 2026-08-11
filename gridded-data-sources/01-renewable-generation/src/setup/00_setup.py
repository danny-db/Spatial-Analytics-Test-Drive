# Databricks notebook source
# Setup & Verification Notebook for Renewable Generation DAB
#
# This notebook initializes the Databricks environment for the renewable energy pipeline:
# 1. Creates the target catalog and schema (idempotent)
# 2. Verifies secrets scope and NREL API key availability
# 3. Tests connectivity to data sources (NASA POWER, NREL NSRDB)
#
# Run this notebook FIRST via the DAB job before the medallion pipeline.
# Safe to re-run — all operations are idempotent.
#
# Prerequisites:
# - Databricks workspace with Unity Catalog enabled
# - Network access to https://power.larc.nasa.gov and https://developer.nrel.gov
#
# Configuration (from DAB variables):
# - catalog: Target Unity Catalog name (default: geo)
# - schema: Target schema name (default: renewable_energy)
# - nrel_secret_scope: Secrets scope for API keys (default: renewable_energy)
# - nrel_secret_key: Secret key name (default: nrel_api_key)

import requests
from datetime import datetime

# ===== READ CONFIGURATION FROM VARIABLES =====
# In DAB context, these are passed via spark.conf
CATALOG = spark.conf.get("spark.databricks.renewable_generation.catalog", "geo")
SCHEMA = spark.conf.get("spark.databricks.renewable_generation.schema", "renewable_energy")
NREL_SECRET_SCOPE = spark.conf.get("spark.databricks.renewable_generation.nrel_secret_scope", "renewable_energy")
NREL_SECRET_KEY = spark.conf.get("spark.databricks.renewable_generation.nrel_secret_key", "nrel_api_key")

print("=" * 80)
print("RENEWABLE ENERGY INGESTION: DAB SETUP & VERIFICATION")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema: {SCHEMA}")
print(f"  Secrets Scope: {NREL_SECRET_SCOPE}")
print(f"  Secrets Key: {NREL_SECRET_KEY}")

# ===== STEP 1: CREATE CATALOG & SCHEMA =====
print("\n" + "=" * 80)
print("STEP 1: Create Catalog & Schema (idempotent)")
print("=" * 80)

try:
    # Create catalog (if it doesn't exist)
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    print(f"✓ Catalog '{CATALOG}' is ready")

    # Create schema (if it doesn't exist)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    print(f"✓ Schema '{CATALOG}.{SCHEMA}' is ready")

    # Verify access
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE {CATALOG}.{SCHEMA}")
    print(f"✓ Both catalog and schema are accessible")

except Exception as e:
    print(f"✗ Error creating/accessing catalog/schema: {str(e)}")
    raise

# ===== STEP 2: VERIFY SECRETS SCOPE =====
print("\n" + "=" * 80)
print("STEP 2: Verify Secrets Scope & NREL API Key")
print("=" * 80)

try:
    # List available secret scopes
    existing_scopes = [s.name for s in dbutils.secrets.listScopes()]

    if NREL_SECRET_SCOPE in existing_scopes:
        print(f"✓ Secrets scope '{NREL_SECRET_SCOPE}' exists")

        # Try to retrieve the NREL API key
        try:
            api_key = dbutils.secrets.get(scope=NREL_SECRET_SCOPE, key=NREL_SECRET_KEY)
            print(f"✓ NREL API key '{NREL_SECRET_KEY}' found in scope '{NREL_SECRET_SCOPE}'")
            print(f"  Key preview: {api_key[:10]}...")
        except Exception as e:
            print(f"⚠ NREL API key '{NREL_SECRET_KEY}' NOT found in scope '{NREL_SECRET_SCOPE}'")
            print(f"  To store your key, run (one-time setup):")
            print(f"  dbutils.secrets.put(scope='{NREL_SECRET_SCOPE}', key='{NREL_SECRET_KEY}', value='YOUR_API_KEY')")
            print(f"  Get a free key at: https://developer.nrel.gov/signup/")
    else:
        print(f"⚠ Secrets scope '{NREL_SECRET_SCOPE}' does NOT exist")
        print(f"  Available scopes: {existing_scopes}")
        print(f"  Create it via Databricks CLI:")
        print(f"  databricks secrets create-scope --scope {NREL_SECRET_SCOPE}")
        print(f"\n  Then store your NREL API key:")
        print(f"  dbutils.secrets.put(scope='{NREL_SECRET_SCOPE}', key='{NREL_SECRET_KEY}', value='YOUR_KEY')")

except Exception as e:
    print(f"⚠ Error managing secrets: {str(e)}")

# ===== STEP 3: TEST DATA SOURCE CONNECTIVITY =====
print("\n" + "=" * 80)
print("STEP 3: Test Data Source Connectivity")
print("=" * 80)

# Test NASA POWER API
print("\n→ Testing NASA POWER API...")
try:
    test_params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": 151.21,  # Sydney
        "latitude": -33.87,
        "start": "20230101",
        "end": "20230102",
        "format": "JSON"
    }
    response = requests.get(
        "https://power.larc.nasa.gov/api/temporal/hourly/point",
        params=test_params,
        timeout=10
    )
    if response.status_code == 200:
        print("✓ NASA POWER API is reachable (HTTP 200)")
        data = response.json()
        props_count = len(data.get('properties', {}).get('parameter', {}))
        print(f"  Sample response: Retrieved {props_count} parameters")
    else:
        print(f"✗ NASA POWER API returned HTTP {response.status_code}")

except Exception as e:
    print(f"✗ Error connecting to NASA POWER: {str(e)}")

# Test NREL NSRDB API
print("\n→ Testing NREL NSRDB API...")
try:
    # Try to retrieve API key from secrets
    api_key_available = False
    try:
        api_key = dbutils.secrets.get(scope=NREL_SECRET_SCOPE, key=NREL_SECRET_KEY)
        api_key_available = True
    except:
        pass

    if api_key_available:
        test_params = {
            "api_key": api_key,
            "full_name": "Databricks_DAB_Setup",
            "email": "setup@databricks.com",
            "affiliation": "Databricks",
            "mailing_list": False,
            "reason": "energy_resource_assessment",
            "dataset": "Himawari",
            "attributes": "ghi",
            "leap_day": False,
            "interval": 60,
            "utc": False,
            "names": "hourly",
            "years": [2023],
            "latitude": -33.87,
            "longitude": 151.21,
        }
        response = requests.get(
            "https://nsrdb.nrel.gov/api/v2/solar/psm3-download",
            params=test_params,
            timeout=30
        )
        if response.status_code == 200:
            print("✓ NREL NSRDB API is reachable and authenticated (HTTP 200)")
        elif response.status_code == 401:
            print("✗ NREL NSRDB API returned HTTP 401 (Unauthorized)")
            print("  → Check your NREL API key in secrets")
        else:
            print(f"✗ NREL NSRDB API returned HTTP {response.status_code}")
    else:
        print("⚠ Skipping NSRDB test (NREL API key not in secrets yet)")
        print("  → Will be required when running the medallion pipeline")

except Exception as e:
    print(f"⚠ Error connecting to NREL NSRDB: {str(e)}")

# ===== STEP 4: SUMMARY & NEXT STEPS =====
print("\n" + "=" * 80)
print("SETUP SUMMARY")
print("=" * 80)

print(f"""
Databricks Environment Ready:
  ✓ Catalog: {CATALOG}
  ✓ Schema: {SCHEMA}
  ✓ Timestamp: {datetime.now().isoformat()}

Next Steps:
  1. (One-time) Create secrets scope if it doesn't exist:
     databricks secrets create-scope --scope {NREL_SECRET_SCOPE}

  2. (One-time) Store your NREL API key:
     dbutils.secrets.put(scope="{NREL_SECRET_SCOPE}", key="{NREL_SECRET_KEY}", value="YOUR_API_KEY")
     Get a free key at: https://developer.nrel.gov/signup/

  3. Run the DAB job to execute the medallion pipeline:
     databricks bundle run renewable_generation_job -t dev

  4. Query results (after pipeline completes):
     SELECT * FROM {CATALOG}.{SCHEMA}.gold_solar_wind_capacity_factor LIMIT 10;

For troubleshooting, see README.md in the bundle root.
""")

print("=" * 80)
print("SETUP COMPLETE")
print("=" * 80)
