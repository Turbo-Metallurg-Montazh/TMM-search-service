import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

import app.state as state
from app.catalogs import list_catalog_files, save_catalog_upload
from app.indexer import build_search_index, load_index_from_disk
from app.search import find_similar_with_prices
from app.spreadsheet_io import export_univer_to_xlsx, import_xlsx_to_univer


REQ = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
LAT = Histogram("http_request_duration_seconds", "Request latency", ["path"])


class SuggestRequest(BaseModel):
    row: list[Any]
    rowIndex: int
    colIndex: int
    topK: int = Field(default=7, ge=1, le=50)


class BuildIndexRequest(BaseModel):
    batch_size: int = Field(default=16, ge=1, le=256)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index_from_disk()
    yield


app = FastAPI(title="EMK Search Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_mw(request, call_next):
    start = time.time()
    resp = await call_next(request)
    duration = time.time() - start
    path = request.url.path
    LAT.labels(path=path).observe(duration)
    REQ.labels(method=request.method, path=path, status=str(resp.status_code)).inc()
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/build-index")
def build_index(req: BuildIndexRequest | None = None):
    batch_size = req.batch_size if req else 16
    try:
        info = build_search_index(batch_size=batch_size)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "index": info}


@app.get("/index-status")
def index_status():
    return {
        "ready": state.INDEX_READY,
        "stale": state.INDEX_STALE,
        "items": len(state.CANDIDATE_NAMES),
        "catalogs": len(list_catalog_files()),
    }


@app.post("/suggest")
def suggest(req: SuggestRequest):
    if not state.INDEX_READY:
        raise HTTPException(status_code=503, detail="Index not built")

    query = str(req.row[0]).strip() if req.row else ""
    if not query:
        return {"options": []}

    matches = find_similar_with_prices(query, top_k=req.topK)
    return {
        "options": [
            {
                "label": f"{m['name']} — {m['price']}",
                "a": m["name"],
                "b": m["price"],
                "score": m["score"],
                "source_file": m.get("source_file"),
                "sheet": m.get("sheet"),
                "row": m.get("row"),
            }
            for m in matches
        ]
    }


@app.get("/catalogs")
def catalogs():
    return {"catalogs": list_catalog_files()}


@app.post("/catalogs/upload")
async def upload_catalog(
    file: UploadFile = File(...),
    rebuild_index: bool = Query(default=False),
    batch_size: int = Query(default=16, ge=1, le=256),
):
    try:
        uploaded = await save_catalog_upload(file)
        if rebuild_index:
            index_info = build_search_index(batch_size=batch_size)
        else:
            state.mark_index_stale()
            index_info = None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "catalog": uploaded,
        "index": index_info,
        "index_ready": state.INDEX_READY,
        "index_stale": state.INDEX_STALE,
    }


@app.post("/catalogs/upload-many")
async def upload_many_catalogs(
    files: list[UploadFile] = File(...),
    rebuild_index: bool = Query(default=False),
    batch_size: int = Query(default=16, ge=1, le=256),
):
    uploaded = []
    try:
        for file in files:
            uploaded.append(await save_catalog_upload(file))
        if rebuild_index:
            index_info = build_search_index(batch_size=batch_size)
        else:
            state.mark_index_stale()
            index_info = None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "catalogs": uploaded,
        "index": index_info,
        "index_ready": state.INDEX_READY,
        "index_stale": state.INDEX_STALE,
    }


@app.post("/import-xlsx")
async def import_xlsx(file: UploadFile = File(...)):
    return await import_xlsx_to_univer(file)


@app.post("/export-xlsx")
def export_xlsx(payload: dict = Body(...)):
    return export_univer_to_xlsx(payload)
