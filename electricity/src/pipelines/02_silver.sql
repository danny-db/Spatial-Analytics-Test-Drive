-- Silver layer — cleaned & typed transmission network + boundaries.
--
-- Drops extraneous columns from the raw source and converts the WKT geometry
-- string into a validated EPSG:4326 geometry text via st_geomfromwkt / st_astext.
-- Migrated from the former 1_dataset/2_Silver_Gold_Layers notebook.

-- Transmission lines (linestrings).
CREATE OR REFRESH MATERIALIZED VIEW silver_transmission_line AS
SELECT
    objectid,
    featuretyp AS feature_type,
    descriptio,
    class,
    name,
    operationa,
    state,
    spatialcon,
    revised,
    ga_guid,
    capacitykv,
    st_lengths,
    length_m,
    st_astext(st_geomfromwkt(geometry)) AS geom_4326
FROM bronze_transmission_line;

-- Major power stations (points).
CREATE OR REFRESH MATERIALIZED VIEW silver_major_power_station AS
SELECT
    objectid,
    featuretyp,
    descriptio,
    class,
    name,
    operationa,
    owner,
    generation,
    primaryfue,
    primarysub,
    generati_1,
    generatorn,
    locality,
    state,
    spatialcon,
    revised,
    comment_,
    ga_guid,
    x_coordina,
    y_coordina,
    st_astext(st_geomfromwkt(geometry)) AS geom_4326
FROM bronze_major_power_station;

-- Transmission substations (points).
CREATE OR REFRESH MATERIALIZED VIEW silver_transmission_substation AS
SELECT
    objectid,
    featuretyp,
    descriptio,
    class,
    name,
    operationa,
    state,
    spatialcon,
    revised,
    ga_guid,
    voltagekv,
    locality,
    comment_,
    x_coordina,
    y_coordina,
    st_astext(st_geomfromwkt(geometry)) AS geom_4326
FROM bronze_transmission_substation;

-- States & territories boundary (multipolygons), used to filter to VIC in gold.
CREATE OR REFRESH MATERIALIZED VIEW geo.digital_boundary.silver_au_states_and_territories AS
SELECT
    STE_NAME21 AS state,
    AUS_NAME21 AS country,
    AREASQKM21 AS area_sqkm,
    st_astext(st_geomfromwkt(geometry)) AS geom_4326
FROM geo.digital_boundary.bronze_au_states_and_territories;
