from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

import app.api as api
import app.state as state
from main import app as main_app


def test_main_exports_fastapi_app():
    assert main_app is api.app


def test_health_is_public_and_does_not_require_ready_index():
    with TestClient(api.app) as client:
        state.reset_index()
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_returns_prometheus_payload():
    with TestClient(api.app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text


def test_strip_api_prefix_middleware_routes_prefixed_health(monkeypatch):
    monkeypatch.setattr(api, "API_PREFIX", "/api")
    with TestClient(api.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_origin_guard_rejects_forbidden_host():
    with TestClient(api.app) as client:
        response = client.get("/catalogs", headers={"host": "evil.example"})

    assert response.status_code == 403
    assert response.text == "Forbidden host"


def test_origin_guard_rejects_forbidden_origin():
    with TestClient(api.app) as client:
        response = client.get("/catalogs", headers={"origin": "https://evil.example"})

    assert response.status_code == 403
    assert response.text == "Forbidden origin"


def test_origin_guard_rejects_forbidden_referer():
    with TestClient(api.app) as client:
        response = client.get("/catalogs", headers={"referer": "https://evil.example/path"})

    assert response.status_code == 403
    assert response.text == "Forbidden referer"


def test_index_status_returns_runtime_state(monkeypatch):
    monkeypatch.setattr(api, "list_catalog_files", lambda: [{"filename": "catalog.xlsx"}])

    with TestClient(api.app) as client:
        state.INDEX_READY = True
        state.INDEX_STALE = True
        state.CANDIDATE_NAMES = ["a", "b"]
        response = client.get("/index-status")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "stale": True, "items": 2, "catalogs": 1}


def test_build_index_uses_default_request(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: calls.append(batch_size) or {"items": 1})

    with TestClient(api.app) as client:
        response = client.post("/build-index")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index": {"items": 1}}
    assert calls == [16]


def test_build_index_validates_batch_size():
    with TestClient(api.app) as client:
        response = client.post("/build-index", json={"batch_size": 0})

    assert response.status_code == 422


def test_build_index_returns_400_for_domain_error(monkeypatch):
    def fail(batch_size):
        raise RuntimeError("No catalogs")

    monkeypatch.setattr(api, "build_search_index", fail)

    with TestClient(api.app) as client:
        response = client.post("/build-index", json={"batch_size": 4})

    assert response.status_code == 400
    assert response.json() == {"detail": "No catalogs"}


def test_suggest_returns_503_when_index_is_not_ready():
    with TestClient(api.app) as client:
        state.reset_index()
        response = client.post("/suggest", json="Автомат")

    assert response.status_code == 503
    assert response.json() == {"detail": "Index not built"}


def test_suggest_returns_empty_for_blank_query():
    with TestClient(api.app) as client:
        state.INDEX_READY = True
        response = client.post("/suggest", json="   ")

    assert response.status_code == 200
    assert response.json() == []


def test_suggest_maps_search_results(monkeypatch):
    state.INDEX_READY = True
    monkeypatch.setattr(
        api,
        "find_similar_with_prices",
        lambda query, top_k: [
            {
                "name": "Автомат",
                "price": 100.0,
                "score": 0.95,
                "source_file": "catalog.xlsx",
                "sheet": "main",
                "row": 2,
            }
        ],
    )

    with TestClient(api.app) as client:
        response = client.post("/suggest", json="автомат")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "Автомат",
            "price": 100.0,
            "similarity": 0.95,
            "source_file": "catalog.xlsx",
            "sheet": "main",
            "row": 2,
        }
    ]


def test_catalogs_returns_file_list(monkeypatch):
    monkeypatch.setattr(api, "list_catalog_files", lambda: [{"filename": "catalog.xlsx"}])

    with TestClient(api.app) as client:
        response = client.get("/catalogs")

    assert response.status_code == 200
    assert response.json() == {"catalogs": [{"filename": "catalog.xlsx"}]}


def test_upload_catalog_without_rebuild_marks_index_stale(monkeypatch):
    async def save(file):
        return {"filename": file.filename, "bytes": 4}

    monkeypatch.setattr(api, "save_catalog_upload", save)

    with TestClient(api.app) as client:
        state.reset_index()
        response = client.post("/catalogs/upload", files={"file": ("a.xlsx", b"data")})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "catalog": {"filename": "a.xlsx", "bytes": 4},
        "index": None,
        "index_ready": False,
        "index_stale": True,
    }


def test_upload_catalog_with_rebuild_returns_index_info(monkeypatch):
    async def save(file):
        return {"filename": file.filename, "bytes": 4}

    monkeypatch.setattr(api, "save_catalog_upload", save)
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: {"items": batch_size})

    with TestClient(api.app) as client:
        response = client.post(
            "/catalogs/upload?rebuild_index=true&batch_size=3",
            files={"file": ("a.xlsx", b"data")},
        )

    assert response.status_code == 200
    assert response.json()["index"] == {"items": 3}


def test_upload_catalog_returns_400_for_validation_error(monkeypatch):
    async def save(file):
        raise ValueError("bad file")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    with TestClient(api.app) as client:
        response = client.post("/catalogs/upload", files={"file": ("a.txt", b"data")})

    assert response.status_code == 400
    assert response.json() == {"detail": "bad file"}


def test_upload_catalog_returns_500_for_unexpected_error(monkeypatch):
    async def save(file):
        raise RuntimeError("disk full")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    with TestClient(api.app) as client:
        response = client.post("/catalogs/upload", files={"file": ("a.xlsx", b"data")})

    assert response.status_code == 500
    assert response.json() == {"detail": "disk full"}


def test_upload_many_catalogs_saves_all_files_and_rebuilds(monkeypatch):
    async def save(file):
        return {"filename": file.filename}

    monkeypatch.setattr(api, "save_catalog_upload", save)
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: {"items": batch_size})

    with TestClient(api.app) as client:
        state.reset_index()
        response = client.post(
            "/catalogs/upload-many?rebuild_index=true&batch_size=2",
            files=[("files", ("a.xlsx", b"a")), ("files", ("b.xlsx", b"b"))],
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "catalogs": [{"filename": "a.xlsx"}, {"filename": "b.xlsx"}],
        "index": {"items": 2},
        "index_ready": False,
        "index_stale": False,
    }


def test_upload_many_catalogs_returns_400_for_validation_error(monkeypatch):
    async def save(file):
        raise ValueError("bad file")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    with TestClient(api.app) as client:
        response = client.post("/catalogs/upload-many", files=[("files", ("a.xlsx", b"a"))])

    assert response.status_code == 400
    assert response.json() == {"detail": "bad file"}


def test_upload_many_catalogs_returns_500_for_unexpected_error(monkeypatch):
    async def save(file):
        raise RuntimeError("disk full")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    with TestClient(api.app) as client:
        response = client.post("/catalogs/upload-many", files=[("files", ("a.xlsx", b"a"))])

    assert response.status_code == 500
    assert response.json() == {"detail": "disk full"}


def test_import_xlsx_delegates_to_converter(monkeypatch):
    async def convert(file):
        return {"filename": file.filename, "sheets": []}

    monkeypatch.setattr(api, "import_xlsx_to_univer", convert)

    with TestClient(api.app) as client:
        response = client.post("/import-xlsx", files={"file": ("a.xlsx", b"data")})

    assert response.status_code == 200
    assert response.json() == {"filename": "a.xlsx", "sheets": []}


def test_export_xlsx_delegates_to_exporter(monkeypatch):
    monkeypatch.setattr(api, "export_univer_to_xlsx", lambda payload: JSONResponse({"received": payload}))

    with TestClient(api.app) as client:
        response = client.post("/export-xlsx", json={"sheets": []})

    assert response.status_code == 200
    assert response.json() == {"received": {"sheets": []}}
