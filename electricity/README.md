# ⚡ Electricity Spatial Medallion

A [Databricks Asset Bundle](https://docs.databricks.com/dev-tools/bundles/index.html)
that builds an Australian electricity-transmission **medallion** (bronze → silver → gold)
in the `geo` Unity Catalog. A single [Lakeflow Declarative Pipeline](https://docs.databricks.com/dlt/index.html)
ingests the source shapefiles, cleans and types the geometries, and produces
Victoria-filtered gold tables — orchestrated by a job with one pipeline task.

## 🗺️ Overview

```
resources/electricity.job.yml   →  job: electricity_medallion_job
resources/electricity.pipeline.yml → pipeline: electricity_medallion
src/pipelines/01_bronze.py      →  geopandas shapefile ingestion (bronze_*)
src/pipelines/02_silver.sql     →  cleaned / typed geometries (silver_*)
src/pipelines/03_gold.sql       →  Victoria-filtered assets (gold_vic_*)
```

### Medallion tables

| Layer  | Table | Schema |
| ------ | ----- | ------ |
| Bronze | `bronze_transmission_line`, `bronze_major_power_station`, `bronze_transmission_substation` | `geo.electricity` |
| Bronze | `bronze_au_states_and_territories`, `bronze_vic_lga` | `geo.digital_boundary` |
| Silver | `silver_transmission_line`, `silver_major_power_station`, `silver_transmission_substation` | `geo.electricity` |
| Silver | `silver_au_states_and_territories` | `geo.digital_boundary` |
| Gold   | `gold_vic_transmission_substation`, `gold_vic_transmission_line` | `geo.electricity` |

The gold layer filters the transmission network to **Victoria** using the VIC state
multipolygon. H3 is used in preference to `st_intersects()` / `st_intersection()` alone
because H3 containment joins parallelise, whereas the ST functions execute on a single
worker. The gold transmission-line table clips lines to the boundary
(containment + intersection) and is currently capped at ≤ 10 segments per line to bound
ST-function cost.

## 🔗 Datasets & references

All datasets are ingested from Unity Catalog volumes that must be pre-populated with the
shapefiles below.

### Digital Atlas of Australia — transmission network
Volume: `/Volumes/geo/electricity/shp`

- [Electricity Transmission Lines](https://digital.atlas.gov.au/datasets/70f23e91102a4d6899a776d093fa08ef_2)
- [Transmission Substations](https://digital.atlas.gov.au/datasets/d5eae2d7c9e54581a5f19d7e95b9883b_0)
- [Major Power Stations](https://digital.atlas.gov.au/datasets/3d0f2d1b8aec4e1a8870b03ce11d4405_1)

### Australian Bureau of Statistics (ABS) — digital boundaries
Volume: `/Volumes/geo/digital_boundary/asgs_ed_3`

- [Australian Statistical Geography Standard (ASGS) Edition 3 — GDA94 digital boundary files](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files#downloads-for-gda94-digital-boundary-files)
- [ASGS Edition 3 structure diagram](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026#asgs-diagram)

### Further reading
- [Introducing Spatial SQL: Databricks' 80+ functions for high-performance geospatial analytics](https://www.databricks.com/blog/introducing-spatial-sql-databricks-80-functions-high-performance-geospatial-analytics) — `ST_DWithin()`, recursive CTEs, and network-tracing patterns for extending the gold layer.
- [About tracing utility networks (ArcGIS Pro)](https://pro.arcgis.com/en/pro-app/latest/help/data/utility-network/about-tracing-utility-networks.htm) — utility-network flow-direction tracing use case.
- [National Electricity Market (NEM) data — AEMO / OpenNEM](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem) — NEM datasets for network-tracing analysis.

## ✅ Prerequisites

- Databricks CLI ≥ v0.230 with a configured profile (`databricks auth login -p <profile>`).
- The `geo` catalog with `electricity` and `digital_boundary` schemas, and the two volumes
  above populated with the shapefiles.
- Serverless compute enabled (the pipeline installs `geopandas` via its environment spec).

## 🚀 Deploy & run

```bash
# From this directory (electricity/):
databricks bundle validate -t dev                       # check config
databricks bundle deploy   -t dev --profile <profile>   # deploy job + pipeline
databricks bundle run electricity_medallion_job -t dev  # run the medallion end-to-end
```

Override any default (catalog, schemas, volume paths) at deploy time, e.g.
`--var="catalog=my_catalog"`. See `databricks.yml` for the full variable list.
