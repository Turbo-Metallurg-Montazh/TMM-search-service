import time
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, Depends # Добавили Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response
from app.services.catalogs.catalogsDelete import DeleteCatalogsRequest
from app.services.catalogs.deleteManyCatalogsService import  delete_many_catalogs_service

import app.state as state
from app.services.catalogs.catalogs import list_catalog_files, save_catalog_upload
from app.config import (
    ALLOWED_HOSTS,
    ALLOWED_ORIGINS,
    ALLOWED_REFERER_ORIGINS,
    API_PREFIX,
    DISABLE_REQUEST_GUARD,
    origin_from_url
)
from app.indexer import build_search_index, load_index_from_disk
from app.search import find_similar_with_prices
from app.spreadsheet_io import export_univer_to_xlsx, import_xlsx_to_univer

from app.dependencies.auth import VerifyScopes

REQ = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
LAT = Histogram("http_request_duration_seconds", "Request latency", ["path"])


class Suggestion(BaseModel):
    name: str
    price: float | None = None
    similarity: float
    source_file: str | None = None
    sheet: str | None = None
    row: int | None = None


class BuildIndexRequest(BaseModel):
    batch_size: int = Field(default=16, ge=1, le=256)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index_from_disk()
    yield


app = FastAPI(title="EMK Search Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def strip_api_prefix_mw(request, call_next):
    if API_PREFIX and request.scope["path"].startswith(f"{API_PREFIX}/"):
        request.scope["root_path"] = API_PREFIX
        request.scope["path"] = request.scope["path"][len(API_PREFIX) :]
    elif API_PREFIX and request.scope["path"] == API_PREFIX:
        request.scope["root_path"] = API_PREFIX
        request.scope["path"] = "/"
    return await call_next(request)


@app.middleware("http")
async def origin_guard_mw(request, call_next):
    if DISABLE_REQUEST_GUARD:
        return await call_next(request)

    health_paths = {"/health"}
    if API_PREFIX:
        health_paths.add(f"{API_PREFIX}/health")
    if request.scope["path"] in health_paths:
        return await call_next(request)

    host = request.headers.get("host", "").split(":", 1)[0].lower()
    if ALLOWED_HOSTS and host and host not in {allowed.lower() for allowed in ALLOWED_HOSTS}:
        return Response("Forbidden host", status_code=403)

    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in ALLOWED_ORIGINS:
        return Response("Forbidden origin", status_code=403)

    referer = request.headers.get("referer")
    referer_origin = origin_from_url(referer) if referer else None
    if referer_origin and referer_origin not in ALLOWED_REFERER_ORIGINS:
        return Response("Forbidden referer", status_code=403)

    return await call_next(request)


@app.middleware("http")
async def metrics_mw(request, call_next):
    start = time.time()
    resp = await call_next(request)
    duration = time.time() - start
    path = request.url.path
    LAT.labels(path=path).observe(duration)
    REQ.labels(method=request.method, path=path, status=str(resp.status_code)).inc()
    return resp


# Публичные эндпоинты
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Защищенные эндпоинты
@app.post("/build-index", dependencies=[Depends(VerifyScopes(["catalogs:write"]))])
def build_index(req: BuildIndexRequest | None = None):
    batch_size = req.batch_size if req else 16
    try:
        info = build_search_index(batch_size=batch_size)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "index": info}


@app.get("/index-status", dependencies=[Depends(VerifyScopes(["catalogs:read"]))])
def index_status():
    return {
        "ready": state.INDEX_READY,
        "stale": state.INDEX_STALE,
        "items": len(state.CANDIDATE_NAMES),
        "catalogs": len(list_catalog_files()),
    }


@app.post("/suggest", response_model=list[Suggestion], dependencies=[Depends(VerifyScopes(["catalogs:read"]))])
def suggest(query: str = Body(..., min_length=1, description="Tender item text to match against supplier catalogs")):
    if not state.INDEX_READY:
        raise HTTPException(status_code=503, detail="Index not built")

    query = query.strip()
    if not query:
        return []

    matches = find_similar_with_prices(query, top_k=3)
    return [
        Suggestion(
            name=m["name"],
            price=m["price"],
            similarity=m["score"],
            source_file=m.get("source_file"),
            sheet=m.get("sheet"),
            row=m.get("row"),
        )
        for m in matches
    ]


@app.get("/catalogs", dependencies=[Depends(VerifyScopes(["catalogs:read"]))])
def catalogs():
    return {"catalogs": list_catalog_files()}


@app.post("/catalogs/upload", dependencies=[Depends(VerifyScopes(["catalogs:write"]))])
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


@app.post("/catalogs/upload-many", dependencies=[Depends(VerifyScopes(["catalogs:write"]))])
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


@app.post("/import-xlsx", dependencies=[Depends(VerifyScopes(["catalogs:write"]))])
async def import_xlsx(file: UploadFile = File(...)):
    return await import_xlsx_to_univer(file)


@app.post("/export-xlsx", dependencies=[Depends(VerifyScopes(["catalogs:read"]))])
def export_xlsx(payload: dict = Body(...)):
    return export_univer_to_xlsx(payload)


@app.post("/catalogs/delete-many", dependencies=[Depends(VerifyScopes(["catalogs:write"]))])
def delete_many_catalogs(req: DeleteCatalogsRequest):
    return delete_many_catalogs_service(req)