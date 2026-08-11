# Databricks notebook source
# Bronze Layer: Solar Irradiance Ingestion (NASA POWER & NREL NSRDB)
#
# This notebook is part of a Databricks Lakeflow Declarative Pipeline (DLT).
# It defines two materialized views as @dp.table functions:
#
# 1. bronze_nasa_power_solar
#    - Fetches hourly solar irradiance from NASA POWER API
#    - Coverage: Australian grid points (representative sample)
#    - Endpoint: https://power.larc.nasa.gov/api/temporal/hourly/point
#    - Variables: ALLSKY_SFC_SW_DWN (GHI), T2M (temperature), WS10M (wind speed at 10m)
#    - Format: NetCDF (parsed with xarray)
#    - No authentication required
#
# 2. bronze_nsrdb_himawari
#    - Fetches hourly solar irradiance from NREL NSRDB Himawari-8/9 satellite
#    - Coverage: Asian-Pacific domain (includes all of Australia), ~2 km resolution
#    - Endpoint: https://nsrdb.nrel.gov/api/v2/solar/psm3-download
#    - Variables: GHI, DNI, DHI, temperature, wind speed
#    - Format: CSV (via NREL API)
#    - Requires NREL Developer API key (free, from https://developer.nrel.gov/signup/)
#
# Configuration (from DAB pipeline configuration):
# - spark.databricks.renewable_generation.start_date: Start date (YYYYMMDD)
# - spark.databricks.renewable_generation.end_date: End date (YYYYMMDD)
# - spark.databricks.renewable_generation.nrel_secret_scope: Secrets scope name
# - spark.databricks.renewable_generation.nrel_secret_key: Secret key name
# - spark.databricks.renewable_generation.lat_min/max, lon_min/max: Bounding box (not yet used; expand grid here)

from pyspark import pipelines as dp
import requests
import io
import pandas as pd
from datetime import datetime
from typing import List, Dict
import xarray as xr

# ===== READ CONFIGURATION =====
START_DATE = spark.conf.get("spark.databricks.renewable_generation.start_date", "20230101")
END_DATE = spark.conf.get("spark.databricks.renewable_generation.end_date", "20231231")
NREL_SECRET_SCOPE = spark.conf.get("spark.databricks.renewable_generation.nrel_secret_scope", "renewable_energy")
NREL_SECRET_KEY = spark.conf.get("spark.databricks.renewable_generation.nrel_secret_key", "nrel_api_key")

print(f"Bronze layer configuration:")
print(f"  Date range: {START_DATE} to {END_DATE}")
print(f"  NREL secret scope: {NREL_SECRET_SCOPE}")
print(f"  NREL secret key: {NREL_SECRET_KEY}")

# ===== AUSTRALIAN REPRESENTATIVE GRID POINTS =====
# For production, expand this list or generate a full grid based on lat/lon bounding box:
# Latitude: -44 to -10, Longitude: 112 to 154
AUS_POINTS = [
    {"name": "Sydney", "lat": -33.87, "lon": 151.21},
    {"name": "Melbourne", "lat": -37.81, "lon": 144.96},
    {"name": "Brisbane", "lat": -27.47, "lon": 153.03},
    {"name": "Perth", "lat": -31.95, "lon": 115.86},
    {"name": "Adelaide", "lat": -34.93, "lon": 138.60},
    {"name": "Darwin", "lat": -12.46, "lon": 130.84},
    {"name": "Hobart", "lat": -42.88, "lon": 147.33},
    {"name": "Canberra", "lat": -35.28, "lon": 149.13},
]

# ===== HELPER FUNCTIONS =====

def fetch_nasa_power_point(lat: float, lon: float, start_date: str, end_date: str, location_name: str = "") -> List[Dict]:
    """
    Fetch hourly solar data from NASA POWER API for a single location.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        location_name: Name of location for tracking

    Returns:
        List of dictionaries with timestamp, lat, lon, and measurements
    """
    NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    PARAMETERS = "ALLSKY_SFC_SW_DWN,T2M,WS10M"  # GHI, Temperature, Wind Speed at 10m

    try:
        params = {
            "parameters": PARAMETERS,
            "community": "RE",  # Renewable Energy community
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "NETCDF"
        }

        response = requests.get(NASA_POWER_API_URL, params=params, timeout=60)
        response.raise_for_status()

        # Parse NetCDF response using xarray
        with xr.open_dataset(io.BytesIO(response.content)) as ds:
            df = ds.to_dataframe().reset_index()

            # Add metadata columns
            df["location_name"] = location_name
            df["data_source"] = "NASA_POWER"
            df["ingestion_timestamp"] = pd.Timestamp.now()

            return df.to_dict("records")

    except Exception as e:
        print(f"Warning: Error fetching NASA POWER data for {location_name} ({lat}, {lon}): {str(e)}")
        return []


def fetch_nsrdb_point(lat: float, lon: float, start_date: str, end_date: str, api_key: str, location_name: str = "") -> List[Dict]:
    """
    Fetch hourly solar data from NREL NSRDB API for a single location.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        api_key: NREL Developer API key
        location_name: Name of location for tracking

    Returns:
        List of dictionaries with timestamp, lat, lon, and measurements
    """
    NSRDB_API_URL = "https://nsrdb.nrel.gov/api/v2/solar/psm3-download"

    try:
        from datetime import datetime as dt
        start_dt = dt.strptime(start_date, "%Y%m%d")
        end_dt = dt.strptime(end_date, "%Y%m%d")

        params = {
            "api_key": api_key,
            "full_name": "Databricks_DLT",
            "email": "data@databricks.com",
            "affiliation": "Databricks",
            "mailing_list": False,
            "reason": "energy_resource_assessment",
            "dataset": "Himawari",  # Asia-Pacific domain (covers all of Australia)
            "attributes": "ghi,dni,dhi,temperature,wind_speed,zenith",
            "leap_day": False,
            "interval": 60,  # Hourly
            "utc": False,
            "names": "hourly",
            "years": [start_dt.year, end_dt.year],
            "latitude": lat,
            "longitude": lon,
        }

        response = requests.get(NSRDB_API_URL, params=params, timeout=120)

        if response.status_code == 200:
            # Parse CSV response
            from io import StringIO
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, skiprows=1)  # Skip NREL header

            # Add metadata columns
            df["location_name"] = location_name
            df["latitude"] = lat
            df["longitude"] = lon
            df["data_source"] = "NSRDB_HIMAWARI"
            df["ingestion_timestamp"] = pd.Timestamp.now()

            return df.to_dict("records")
        else:
            print(f"Warning: NREL NSRDB API error for {location_name}: HTTP {response.status_code}")
            return []

    except Exception as e:
        print(f"Warning: Error fetching NSRDB data for {location_name} ({lat}, {lon}): {str(e)}")
        return []


# ===== DLT TABLE DEFINITIONS =====

@dp.table(
    name="bronze_nasa_power_solar",
    comment="Raw hourly solar irradiance data from NASA POWER API (GHI, DNI, DHI, temperature, wind speed)",
    table_properties={
        "owner": "data_engineering",
        "data_source": "NASA POWER",
        "refresh_interval": "1 day"
    }
)
def bronze_nasa_power():
    """
    Fetch and return hourly solar data from NASA POWER API.

    NASA POWER provides global gridded meteorological and solar irradiance data
    at ~0.5° × 0.625° resolution (MERRA-2 / satellite-based).

    Australian coverage: lat -44 to -10, lon 112 to 154 (representative grid points)

    Variables:
    - ALLSKY_SFC_SW_DWN: All-sky surface shortwave downward irradiance (W/m²) [GHI]
    - T2M: Temperature at 2 m (K)
    - WS10M: Wind speed at 10 m (m/s)

    Date range: ${START_DATE} to ${END_DATE}
    """
    all_records = []

    print(f"Fetching NASA POWER data for {len(AUS_POINTS)} locations...")

    for point in AUS_POINTS:
        records = fetch_nasa_power_point(
            lat=point["lat"],
            lon=point["lon"],
            start_date=START_DATE,
            end_date=END_DATE,
            location_name=point["name"]
        )
        all_records.extend(records)
        print(f"  {point['name']}: {len(records)} records")

    print(f"Total NASA POWER records: {len(all_records)}")

    if all_records:
        pdf = pd.DataFrame(all_records)
        return spark.createDataFrame(pdf)
    else:
        # Return empty DataFrame with expected schema if no data fetched
        empty_schema = "longitude DOUBLE, latitude DOUBLE, time STRING, ALLSKY_SFC_SW_DWN DOUBLE, T2M DOUBLE, WS10M DOUBLE, location_name STRING, data_source STRING, ingestion_timestamp TIMESTAMP"
        return spark.createDataFrame([], schema=empty_schema)


@dp.table(
    name="bronze_nsrdb_himawari",
    comment="Raw hourly solar irradiance data from NREL NSRDB Himawari-8/9 satellite (GHI, DNI, DHI, temperature, wind speed)",
    table_properties={
        "owner": "data_engineering",
        "data_source": "NREL NSRDB Himawari",
        "refresh_interval": "1 day"
    }
)
def bronze_nsrdb_himawari():
    """
    Fetch and return hourly solar data from NREL NSRDB Himawari API.

    NREL NSRDB provides satellite-based solar irradiance data for the Asia-Pacific domain,
    covering all of Australia at ~2 km resolution, 10-minute to hourly, 2011 onward.

    Australian coverage: lat -44 to -10, lon 112 to 154 (representative grid points)

    Variables:
    - GHI, DNI, DHI: Global/Direct/Diffuse horizontal irradiance (W/m²)
    - Temperature: Air temperature (°C or K depending on dataset)
    - Wind Speed: Wind speed at measurement height (m/s)
    - Solar Zenith: Solar zenith angle (degrees)

    Date range: ${START_DATE} to ${END_DATE}

    Requires NREL Developer API key (free account at https://developer.nrel.gov/signup/).
    Key stored in Databricks secrets scope: ${NREL_SECRET_SCOPE}/${NREL_SECRET_KEY}
    """

    # Retrieve API key from Databricks secrets
    try:
        api_key = dbutils.secrets.get(scope=NREL_SECRET_SCOPE, key=NREL_SECRET_KEY)
    except Exception as e:
        error_msg = f"Error retrieving NREL API key from secrets: {str(e)}"
        print(f"⚠ {error_msg}")
        print(f"   Set up your API key with: dbutils.secrets.put(scope='{NREL_SECRET_SCOPE}', key='{NREL_SECRET_KEY}', value='YOUR_KEY')")
        print(f"   Get a free key at: https://developer.nrel.gov/signup/")
        raise RuntimeError(error_msg)

    all_records = []

    print(f"Fetching NREL NSRDB Himawari data for {len(AUS_POINTS)} locations...")

    for point in AUS_POINTS:
        records = fetch_nsrdb_point(
            lat=point["lat"],
            lon=point["lon"],
            start_date=START_DATE,
            end_date=END_DATE,
            api_key=api_key,
            location_name=point["name"]
        )
        all_records.extend(records)
        print(f"  {point['name']}: {len(records)} records")

    print(f"Total NSRDB Himawari records: {len(all_records)}")

    if all_records:
        pdf = pd.DataFrame(all_records)
        return spark.createDataFrame(pdf)
    else:
        # Return empty DataFrame with expected schema if no data fetched
        # Note: NSRDB CSV columns may vary; adjust as needed based on actual response
        empty_schema = "Year INT, Month INT, Day INT, Hour INT, GHI DOUBLE, DNI DOUBLE, DHI DOUBLE, Temperature DOUBLE, Wind Speed DOUBLE, Solar Zenith DOUBLE, latitude DOUBLE, longitude DOUBLE, location_name STRING, data_source STRING, ingestion_timestamp TIMESTAMP"
        return spark.createDataFrame([], schema=empty_schema)
