# Bronze layer — raw shapefile ingestion.
#
# Each shapefile is read with geopandas, its geometry serialised to WKT, and
# materialised as a bronze table. Datasets are sourced from the Digital Atlas of
# Australia (transmission network) and the ABS ASGS Edition 3 digital boundary
# files. See ../../README.md for dataset links and provenance.
#
# Migrated from the former 1_dataset/1_Load_Bronze_Layer notebook. Volume paths
# are supplied via pipeline configuration rather than notebook globals.

import dlt
import geopandas as gpd
import pandas as pd

# Volume locations, injected from the pipeline `configuration` block.
ELEC_VOLUME = spark.conf.get("electricity.shp_volume")
BOUND_VOLUME = spark.conf.get("electricity.boundary_volume")


def _read_shapefile_to_wkt(volume_loc: str, file_loc: str):
    """Read a shapefile, convert geometry to WKT, return a Spark DataFrame."""
    gdf = gpd.read_file(f"{volume_loc}/{file_loc}")
    gdf["geometry"] = gdf["geometry"].to_wkt()
    pdf = pd.DataFrame(gdf)
    return spark.createDataFrame(pdf)


# --- Transmission network (schema: geo.electricity) ---------------------------

@dlt.table(name="bronze_transmission_line", comment="Raw electricity transmission lines (Digital Atlas of Australia).")
def bronze_transmission_line():
    return _read_shapefile_to_wkt(ELEC_VOLUME, "Electricity_Transmission_Lines/Electricity_Transmission_Lines.shp")


@dlt.table(name="bronze_major_power_station", comment="Raw major power stations (Digital Atlas of Australia).")
def bronze_major_power_station():
    return _read_shapefile_to_wkt(ELEC_VOLUME, "Major_Power_Stations/Major_Power_Stations.shp")


@dlt.table(name="bronze_transmission_substation", comment="Raw transmission substations (Digital Atlas of Australia).")
def bronze_transmission_substation():
    return _read_shapefile_to_wkt(ELEC_VOLUME, "Transmission_Substations/Transmission_Substations.shp")


# --- Digital boundaries (schema: geo.digital_boundary) ------------------------
# Fully-qualified names publish these into the digital_boundary schema alongside
# the electricity schema used for the transmission tables above.

@dlt.table(name="geo.digital_boundary.bronze_au_states_and_territories", comment="Raw ASGS states & territories boundaries (ABS ASGS Edition 3).")
def bronze_au_states_and_territories():
    return _read_shapefile_to_wkt(BOUND_VOLUME, "States and Territories_2021_AUST_SHP_GDA94/STE_2021_AUST_GDA94.shp")


# Ingested for parity with the source notebook; not consumed by silver/gold yet.
@dlt.table(name="geo.digital_boundary.bronze_vic_lga", comment="Raw ASGS Local Government Areas (ABS ASGS Edition 3). Not yet consumed downstream.")
def bronze_vic_lga():
    return _read_shapefile_to_wkt(BOUND_VOLUME, "LGA_2025_AUST_GDA2020/LGA_2025_AUST_GDA2020.shp")
