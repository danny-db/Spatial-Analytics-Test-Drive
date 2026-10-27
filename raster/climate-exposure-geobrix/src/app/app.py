"""Climate Exposure Explorer: Databricks App backend.

FastAPI serves a Three.js front end (static/) plus a small JSON API over the gold tables built by the
GeoBrix notebooks. Queries run on a SQL warehouse via the Statement Execution API, using the app's
service principal. Catalog, schema and warehouse come from the bundle (resources/climate_exposure.app.yml). Results are cached in memory: the data is static per build.
"""
import os
import time
from functools import lru_cache

import h3
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Disposition, Format, StatementParameterListItem, StatementState
from fastapi import FastAPI, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

CATALOG = os.getenv("CLIMATE_CATALOG", "main")
SCHEMA = os.getenv("CLIMATE_SCHEMA", "climate_exposure")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
FQ = f"{CATALOG}.{SCHEMA}"

w = WorkspaceClient()
app = FastAPI(title="Climate Exposure Explorer")
app.add_middleware(GZipMiddleware, minimum_size=1000)
STATIC = os.path.join(os.path.dirname(__file__), "static")


def sql(statement: str, params: list | None = None) -> list[dict]:
    """Run SQL on the warehouse and return rows as dicts (follows result chunks)."""
    resp = w.statement_execution.execute_statement(
        statement=statement, warehouse_id=WAREHOUSE_ID, wait_timeout="50s",
        disposition=Disposition.INLINE, format=Format.JSON_ARRAY,
        parameters=[StatementParameterListItem(**p) for p in (params or [])])
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    cols = [c.name for c in resp.manifest.schema.columns]
    rows = list(resp.result.data_array or [])
    chunk = resp.result
    while chunk.next_chunk_index is not None:
        chunk = w.statement_execution.get_statement_result_chunk_n(resp.statement_id, chunk.next_chunk_index)
        rows.extend(chunk.data_array or [])
    return [dict(zip(cols, r)) for r in rows]


def num(v):
    return None if v is None else float(v)


@lru_cache(maxsize=1)
def cells_payload() -> dict:
    """All land cells + columnar metric layers for every scenario/period, so the UI can morph instantly."""
    rows = sql(f"""SELECT cell_id, state, lat, lon, population, scenario, period,
                          tasmax_ann, tasmax_djf, hot_months, pr_ann_mm,
                          d_tasmax_ann, d_tasmax_djf, d_hot_months, pct_pr
                   FROM {FQ}.gold_h3_climate ORDER BY cell_id""")
    metrics = ["tasmax_ann", "tasmax_djf", "hot_months", "pr_ann_mm", "d_tasmax_ann", "d_tasmax_djf", "d_hot_months", "pct_pr"]
    index, cells, layers = {}, [], {}
    for r in rows:
        cid = int(r["cell_id"])
        if cid not in index:
            index[cid] = len(cells)
            boundary = h3.cell_to_boundary(h3.int_to_str(cid))          # [(lat, lon), ...]
            cells.append({"id": h3.int_to_str(cid), "state": r["state"], "lat": num(r["lat"]), "lon": num(r["lon"]),
                          "pop": int(r["population"] or 0), "b": [[round(a, 4), round(o, 4)] for a, o in boundary]})
    n = len(cells)
    for r in rows:
        key = f"{r['scenario']}|{r['period']}"
        layer = layers.setdefault(key, {m: [None] * n for m in metrics})
        i = index[int(r["cell_id"])]
        for m in metrics:
            layer[m][i] = num(r[m])
    return {"cells": cells, "layers": layers, "source": f"{FQ}.gold_h3_climate"}


@app.get("/api/cells")
def api_cells():
    return cells_payload()


@lru_cache(maxsize=1)
def summary_payload() -> list[dict]:
    return sql(f"SELECT * FROM {FQ}.gold_exposure_summary ORDER BY state, period, scenario")


@app.get("/api/summary")
def api_summary():
    return summary_payload()


@lru_cache(maxsize=1)
def outlines_payload() -> list[dict]:
    return sql(f"SELECT state, geojson FROM {FQ}.gold_state_outlines")


@app.get("/api/outlines")
def api_outlines():
    return outlines_payload()


@lru_cache(maxsize=64)
def heat_exposure(scenario: str, period: str, min_warming: float) -> list[dict]:
    # Calls the governed UC table function built in notebook 03: the same logic Genie, BI and ArcGIS (ODBC) can call
    return sql(f"SELECT * FROM {FQ}.heat_exposure(:scenario, :period, :min_warming) ORDER BY population DESC",
               [{"name": "scenario", "value": scenario}, {"name": "period", "value": period},
                {"name": "min_warming", "value": str(min_warming), "type": "DOUBLE"}])


@app.get("/api/heat_exposure")
def api_heat_exposure(scenario: str = Query("ssp370"), period: str = Query("2091-2100"), min_warming: float = Query(3.0)):
    return {"function": f"{FQ}.heat_exposure", "rows": heat_exposure(scenario, period, round(min_warming, 1))}


@lru_cache(maxsize=64)
def top_sa2(scenario: str, period: str) -> list[dict]:
    return sql(f"""SELECT sa2_name, state, population, tasmax_djf, d_tasmax_djf, hot_months
                   FROM {FQ}.gold_sa2_exposure WHERE scenario = :scenario AND period = :period
                   ORDER BY population * d_tasmax_djf DESC LIMIT 8""",
               [{"name": "scenario", "value": scenario}, {"name": "period", "value": period}])


@app.get("/api/top_sa2")
def api_top_sa2(scenario: str = Query("ssp370"), period: str = Query("2091-2100")):
    return top_sa2(scenario, period)


@app.get("/api/health")
def health():
    return {"ok": True, "warehouse": WAREHOUSE_ID, "schema": FQ}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
