from __future__ import annotations
import time
import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

import app.api as api
import app.state as state
from main import app as main_app
from app.services.catalogs import catalogsDelete
from app.dependencies.auth import JWTBearer, VerifyScopes
from app.dependencies.auth import jwt_bearer, VerifyScopes

# ==============================================================================
# ФИКСТУРЫ
# ==============================================================================

@pytest.fixture
def client(monkeypatch):
    """
    Клиент со сквозной авторизацией (mock).
    Используется для тестирования основной бизнес-логики эндпоинтов.
    """
    monkeypatch.setattr(api, "load_index_from_disk", lambda: False)

    fake_token_payload = {
        "sub": "test_user",
        "scopes": ["admin", "read", "write", "catalogs:write", "catalogs:read"]
    }

    # Подменяем вызовы декораторов
    monkeypatch.setattr(JWTBearer, "__call__", lambda self, credentials=None: fake_token_payload)
    monkeypatch.setattr(VerifyScopes, "__call__", lambda self, payload=None: fake_token_payload)

    # Переопределяем зависимости в FastAPI
    for app_instance in [main_app, api.app]:
        app_instance.dependency_overrides[JWTBearer] = lambda: fake_token_payload
        app_instance.dependency_overrides[VerifyScopes] = lambda: fake_token_payload

    with TestClient(main_app) as test_client:
        state.reset_index()
        yield test_client
        state.reset_index()

    main_app.dependency_overrides.clear()
    api.app.dependency_overrides.clear()


@pytest.fixture
def clean_client(monkeypatch):
    """
    Чистый клиент БЕЗ подмены авторизации.
    Используется исключительно для тестирования middleware, JWTBearer и VerifyScopes.
    """
    monkeypatch.setattr(api, "load_index_from_disk", lambda: False)
    with TestClient(main_app) as test_client:
        yield test_client


# ==============================================================================
# ТЕСТЫ СТРУКТУРЫ И MIDDLEWARE (ОБЩИЕ)
# ==============================================================================

def test_main_exports_fastapi_app():
    assert main_app is api.app


def test_health_is_public_and_does_not_require_ready_index(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_returns_prometheus_payload(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text


def test_strip_api_prefix_middleware_routes_prefixed_health(monkeypatch, client):
    monkeypatch.setattr(api, "API_PREFIX", "/api")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_origin_guard_rejects_forbidden_host(client):
    response = client.get("/catalogs", headers={"host": "evil.example"})
    assert response.status_code == 403
    assert response.text == "Forbidden host"


def test_request_guard_can_be_disabled_for_local_clients(monkeypatch, client):
    monkeypatch.setattr(api, "DISABLE_REQUEST_GUARD", True)
    monkeypatch.setattr(api, "list_catalog_files", lambda: [])

    response = client.get("/catalogs", headers={"host": "evil.example"})
    assert response.status_code == 200
    assert response.json() == {"catalogs": []}


def test_origin_guard_rejects_forbidden_origin(client):
    response = client.get("/catalogs", headers={"origin": "https://evil.example"})
    assert response.status_code == 403
    assert response.text == "Forbidden origin"


def test_origin_guard_rejects_forbidden_referer(client):
    response = client.get("/catalogs", headers={"referer": "https://evil.example/path"})
    assert response.status_code == 403
    assert response.text == "Forbidden referer"


# ==============================================================================
# ТЕСТЫ ЭНДПОИНТОВ И ИХ ВАЛИДАЦИИ
# ==============================================================================

def test_index_status_returns_runtime_state(monkeypatch, client):
    monkeypatch.setattr(api, "list_catalog_files", lambda: [{"filename": "catalog.xlsx"}])

    state.INDEX_READY = True
    state.INDEX_STALE = True
    state.CANDIDATE_NAMES = ["a", "b"]
    response = client.get("/index-status")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "stale": True, "items": 2, "catalogs": 1}


def test_build_index_uses_default_request(monkeypatch, client):
    calls = []
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: calls.append(batch_size) or {"items": 1})

    response = client.post("/build-index")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index": {"items": 1}}
    assert calls == [16]


@pytest.mark.parametrize("batch_size", [0, 257])
def test_build_index_validates_batch_size_boundaries(client, batch_size):
    """Проверяет ограничения ge=1 и le=256 для схемы Pydantic."""
    response = client.post("/build-index", json={"batch_size": batch_size})
    assert response.status_code == 422


def test_build_index_returns_400_for_domain_error(monkeypatch, client):
    def fail(batch_size):
        raise RuntimeError("No catalogs")

    monkeypatch.setattr(api, "build_search_index", fail)

    response = client.post("/build-index", json={"batch_size": 4})
    assert response.status_code == 400
    assert response.json() == {"detail": "No catalogs"}


def test_suggest_returns_503_when_index_is_not_ready(client):
    response = client.post("/suggest", json="Автомат")
    assert response.status_code == 503
    assert response.json() == {"detail": "Index not built"}


def test_suggest_returns_empty_for_blank_query(client):
    state.INDEX_READY = True
    response = client.post("/suggest", json="   ")
    assert response.status_code == 200
    assert response.json() == []


def test_suggest_requires_min_length_validation(client):
    """Проверяет валидацию min_length=1 на уровне FastAPI Body."""
    state.INDEX_READY = True
    response = client.post("/suggest", json="")
    assert response.status_code == 422


def test_suggest_maps_search_results(monkeypatch, client):
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


def test_catalogs_returns_file_list(monkeypatch, client):
    monkeypatch.setattr(api, "list_catalog_files", lambda: [{"filename": "catalog.xlsx"}])

    response = client.get("/catalogs")
    assert response.status_code == 200
    assert response.json() == {"catalogs": [{"filename": "catalog.xlsx"}]}


# --- ТЕСТЫ ЗАГРУЗКИ / УДАЛЕНИЯ КАТАЛОГОВ ---

@pytest.mark.parametrize("endpoint", ["/catalogs/upload", "/catalogs/upload-many"])
@pytest.mark.parametrize("batch_size", [0, 257])
def test_upload_endpoints_validate_batch_size_query(client, endpoint, batch_size):
    """Проверяет валидацию Query параметров ge=1 и le=256 при загрузке."""
    files = {"file": ("a.xlsx", b"data")} if endpoint == "/catalogs/upload" else [("files", ("a.xlsx", b"data"))]
    response = client.post(f"{endpoint}?batch_size={batch_size}", files=files)
    assert response.status_code == 422


def test_upload_catalog_without_rebuild_marks_index_stale(monkeypatch, client):
    async def save(file):
        return {"filename": file.filename, "bytes": 4}

    monkeypatch.setattr(api, "save_catalog_upload", save)

    response = client.post("/catalogs/upload", files={"file": ("a.xlsx", b"data")})
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "catalog": {"filename": "a.xlsx", "bytes": 4},
        "index": None,
        "index_ready": False,
        "index_stale": True,
    }


def test_upload_catalog_with_rebuild_returns_index_info(monkeypatch, client):
    async def save(file):
        return {"filename": file.filename, "bytes": 4}

    monkeypatch.setattr(api, "save_catalog_upload", save)
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: {"items": batch_size})

    response = client.post(
        "/catalogs/upload?rebuild_index=true&batch_size=3",
        files={"file": ("a.xlsx", b"data")},
    )
    assert response.status_code == 200
    assert response.json()["index"] == {"items": 3}


def test_upload_catalog_returns_400_for_validation_error(monkeypatch, client):
    async def save(file):
        raise ValueError("bad file")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    response = client.post("/catalogs/upload", files={"file": ("a.txt", b"data")})
    assert response.status_code == 400
    assert response.json() == {"detail": "bad file"}


def test_upload_catalog_returns_500_for_unexpected_error(monkeypatch, client):
    async def save(file):
        raise RuntimeError("disk full")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    response = client.post("/catalogs/upload", files={"file": ("a.xlsx", b"data")})
    assert response.status_code == 500
    assert response.json() == {"detail": "disk full"}


def test_upload_many_catalogs_saves_all_files_and_rebuilds(monkeypatch, client):
    async def save(file):
        return {"filename": file.filename}

    monkeypatch.setattr(api, "save_catalog_upload", save)
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: {"items": batch_size})

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


def test_upload_many_catalogs_returns_400_for_validation_error(monkeypatch, client):
    async def save(file):
        raise ValueError("bad file")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    response = client.post("/catalogs/upload-many", files=[("files", ("a.xlsx", b"a"))])
    assert response.status_code == 400
    assert response.json() == {"detail": "bad file"}


def test_upload_many_catalogs_returns_500_for_unexpected_error(monkeypatch, client):
    async def save(file):
        raise RuntimeError("disk full")

    monkeypatch.setattr(api, "save_catalog_upload", save)

    response = client.post("/catalogs/upload-many", files=[("files", ("a.xlsx", b"a"))])
    assert response.status_code == 500
    assert response.json() == {"detail": "disk full"}


def test_import_xlsx_delegates_to_converter(monkeypatch, client):
    async def convert(file):
        return {"filename": file.filename, "sheets": []}

    monkeypatch.setattr(api, "import_xlsx_to_univer", convert)

    response = client.post("/import-xlsx", files={"file": ("a.xlsx", b"data")})
    assert response.status_code == 200
    assert response.json() == {"filename": "a.xlsx", "sheets": []}


def test_export_xlsx_delegates_to_exporter(monkeypatch, client):
    monkeypatch.setattr(api, "export_univer_to_xlsx", lambda payload: JSONResponse({"received": payload}))

    response = client.post("/export-xlsx", json={"sheets": []})
    assert response.status_code == 200
    assert response.json() == {"received": {"sheets": []}}


def test_delete_catalog_success(monkeypatch, client, tmp_path):
    catalog = tmp_path / "catalog.xlsx"
    catalog.write_text("test")
    monkeypatch.setattr(catalogsDelete, "PRICE_LIST_DIR", tmp_path)
    state.INDEX_STALE = False

    response = client.post(
        "/catalogs/delete-many",
        json={"files": [{"filename": "catalog", "extension": "xlsx"}]}
    )
    assert response.status_code == 200
    assert response.json() == {
        "Status": "ok",
        "deletedCount": 1,
        "deletedFiles": ["catalog.xlsx"]
    }
    assert not catalog.exists()
    assert state.INDEX_STALE is True


def test_delete_catalog_missing_file(monkeypatch, client, tmp_path):
    monkeypatch.setattr(catalogsDelete, "PRICE_LIST_DIR", tmp_path)
    response = client.post(
        "/catalogs/delete-many",
        json={"files": [{"filename": "missing", "extension": "xlsx"}]}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "File not found: missing.xlsx"}


def test_delete_catalog_path_traversal(client, monkeypatch, tmp_path):
    monkeypatch.setattr(catalogsDelete, "PRICE_LIST_DIR", tmp_path)

    response = client.post(
        "/catalogs/delete-many",
        json={"files": [{"filename": "../secret", "extension": "xlsx"}]}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid filename"}


# ==============================================================================
# ТЕСТЫ БЕЗОПАСНОСТИ (JWT BEARER & SCOPES)
# ==============================================================================

def test_jwt_bearer_missing_credentials(clean_client):
    """Проверяет, что защищенный эндпоинт отклоняет запросы без токена."""
    response = clean_client.get("/catalogs")
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


def test_jwt_bearer_invalid_scheme(clean_client):
    """Проверяет отклонение схем авторизации, отличных от Bearer."""
    headers = {"Authorization": "Basic bG9naW46cGFzc3dvcmQ="}
    response = clean_client.get("/catalogs", headers=headers)
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


def test_verify_scopes_insufficient_permissions(clean_client):
    """Проверяет работу фабрики VerifyScopes при нехватке прав."""
    fake_read_only_payload = {
        "sub": "reader_user",
        "scopes": ["catalogs:read"]
    }

    for app_instance in [main_app, api.app]:
        app_instance.dependency_overrides[jwt_bearer] = lambda: fake_read_only_payload

    try:
        response = clean_client.post("/build-index", json={"batch_size": 16})

        assert response.status_code == 403
        assert "Insufficient permissions" in response.json()["detail"]
    finally:
        main_app.dependency_overrides.clear()
        api.app.dependency_overrides.clear()


# ==============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ С НАСТОЯЩИМ RS256 JWT
# ==============================================================================

RSA_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAmkwbWMcIIsVsYSJut+1Y4EeI7vjI/It9BbiWsNhJVRQqSPed
dDRa6lWIVuU3tmiylecyb5a+DGXaR6u/7OeGs51ppjAtOa1AglIHGkYyVhGU2vAV
PeIOeporbD+qaTefljyM+cQ2hLhtbjXfgkK2fSRDa2ODVLLd6Vj3pkfngv3Qd794
Fb1rPnmWm30XEszYbUxKSNQxlVM705bp7DezneCxFssCzbd4G6Sl0YaFD+617KQr
L0Pi/G5zt3q6obMwLQtERRq9ekwA3bBHOkBiPhyx53mKKCp0Pe3OBRPGGJfkyu/C
QO5vt/zWCE7xx0o4luRRJ+fNuNeAtPXjUkbVewIDAQABAoIBAGq3Pqa9KYxa/SWF
WgxN1Q1xjGyzltbMZtDhJR/0x2tXghNrZvQrDcJLG/v1lv3LFdEF0WVKXfFXNIwj
Zp+kVg6+TYbKhU0B3b8EmrL6X/AVQt3V9OsTAS6cmHHK6sLQ3Mhc4qgQpCBRKiZy
jj6ag3qz7QlGD1wyRbcReF8CFkYK0QH9avkDQNzjLpTYoNAetvRL095tCJwbd0sh
930DCsNLDyS+TaEq3her9BSlkmaHQNI3sPd0KTJ7OnYQWdmu6+f1c3+hwjKroSql
308aFE//yc4T6GMgqDM4DXM46VoBZWVoHI40IMWp06K70UO3sW9Bu23/di/IT0I0
p4IucnkCgYEAyhM4+GTgoCGardghZKRwixN4izH0spBZl+ojoMKbt3eSp1JN+vOU
2Ag+MUwKDH2MX8jpiPhGx8G6Vi7Y7zR8xBNIhWgXuZmZPMIwP84aqaY2qktZxK2/
VBwTt/JD2BwKVmr+3OZUAeHQkAygKT9XilUmcWLBQHyVd2UDIM6P9y0CgYEAw3jw
oLHII9KeRFfChg0+KjFmrJ0iXF/u7gCtr8op0KL/wDjtdGJWAJa8KaONOlZttCDe
SwoZUE3dRBKu9GEDt9yNkBZer7MkzAkSKtQ2X82Xkihw3E59P2NcH8fUUJTcIeRb
ie4wXkW27SUc4Hgg6zP4pyywA30Xqn5KE8ELaEcCgYB467HOqgbkq9c0qj2pTOFv
x9H9cYJdDBYg2uJBA9NMoUfnyk+RmQr7j0swErF8sfA7LS3aYb9xL5NCmTwFQCJc
7rEZ66Uu0iQpgIaA1+OKm0Tg+MAZ+mKggUCndVh1zKm+9r3WEBo7GhbE7Fk29Yl7
5OJhPVgpL6P9Uzvg+NqbbQKBgEJRlvo+NxQIUlAHomTOu2efSSGJUm4a0jqHmmYI
5fT2SGUUK2QQNPOQMJjD95dyWVgCysiUzY/USxzcZeVdwOAxgQoAvPFJi1N6RGKp
iyUn4KPi+p+UNaQ69reFmcAZMTKCgpgiauChMHX24Hw75ZdHE7bMT49vcocSv9lB
5rfjAoGAQ91GcuPXG/eyy+etEhCDR9g0XN6eBzXEG0VQst5Dh9FtCeO4YeSS4wXL
4qkfrFl/VouQ/uxnmGSZjNIMxsjYE1tEu0FH3FhaZBT6FciixjpoLGYif8JnmuG1
3YbQBjSDl7dLVqNSAkyIRsnlOLkVV0P927DKNyATk775Ggt+d3Y=
-----END RSA PRIVATE KEY-----"""

RSA_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAmkwbWMcIIsVsYSJut+1Y
4EeI7vjI/It9BbiWsNhJVRQqSPeddDRa6lWIVuU3tmiylecyb5a+DGXaR6u/7OeG
s51ppjAtOa1AglIHGkYyVhGU2vAVPeIOeporbD+qaTefljyM+cQ2hLhtbjXfgkK2
fSRDa2ODVLLd6Vj3pkfngv3Qd794Fb1rPnmWm30XEszYbUxKSNQxlVM705bp7Dez
neCxFssCzbd4G6Sl0YaFD+617KQrL0Pi/G5zt3q6obMwLQtERRq9ekwA3bBHOkBi
Phyx53mKKCp0Pe3OBRPGGJfkyu/CQO5vt/zWCE7xx0o4luRRJ+fNuNeAtPXjUkbV
ewIDAQAB
-----END PUBLIC KEY-----"""


def _make_token(scopes: list[str], exp_offset: int = 600, sub: str = "emk-backend"):
    import jwt
    now = int(time.time())
    payload = {"sub": sub, "scopes": scopes, "iat": now, "exp": now + exp_offset}
    return jwt.encode(payload, RSA_PRIVATE_KEY, algorithm="RS256")


@pytest.fixture
def rs256_client(monkeypatch, tmp_path):
    """Клиент с реальной проверкой RS256 JWT (без моков авторизации)."""
    import app.config as config
    import app.dependencies.auth as auth_mod

    monkeypatch.setattr(api, "load_index_from_disk", lambda: False)
    monkeypatch.setattr(config, "JWT_PUBLIC_KEY", RSA_PUBLIC_KEY)
    monkeypatch.setattr(auth_mod, "JWT_PUBLIC_KEY", RSA_PUBLIC_KEY)

    for app_instance in [main_app, api.app]:
        app_instance.dependency_overrides.clear()

    with TestClient(main_app) as test_client:
        state.reset_index()
        yield test_client
        state.reset_index()

    main_app.dependency_overrides.clear()
    api.app.dependency_overrides.clear()


def test_rs256_valid_token_with_read_scopes(monkeypatch, rs256_client):
    monkeypatch.setattr(api, "list_catalog_files", lambda: [{"filename": "cat.xlsx"}])

    token = _make_token(["catalogs:read"])
    response = rs256_client.get("/catalogs", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {"catalogs": [{"filename": "cat.xlsx"}]}


def test_rs256_valid_token_with_write_scopes(monkeypatch, rs256_client):
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: {"items": 1})

    token = _make_token(["catalogs:write"])
    response = rs256_client.post(
        "/build-index",
        json={"batch_size": 16},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index": {"items": 1}}


def test_rs256_rejects_token_with_wrong_scopes(monkeypatch, rs256_client):
    monkeypatch.setattr(api, "build_search_index", lambda batch_size: {"items": 1})

    token = _make_token(["catalogs:read"])
    response = rs256_client.post(
        "/build-index",
        json={"batch_size": 16},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    assert "Insufficient permissions" in response.json()["detail"]


def test_rs256_rejects_expired_token(monkeypatch, rs256_client):
    monkeypatch.setattr(api, "list_catalog_files", lambda: [])

    token = _make_token(["catalogs:read"], exp_offset=-10)
    response = rs256_client.get("/catalogs", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_rs256_rejects_token_with_invalid_signature(monkeypatch, rs256_client):
    import jwt as pyjwt
    monkeypatch.setattr(api, "list_catalog_files", lambda: [])

    now = int(time.time())
    payload = {"sub": "attacker", "scopes": ["catalogs:read"], "iat": now, "exp": now + 600}
    wrong_key = """-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDea8Fmxb2ltcL6
ySRGvmCpUfGj5PWIMAOzHklAftP4U9Ot5nRP6MJdN5eCO6Ldal69cUc47k9jZHuW
P2W8WIQfjoQ9tii3l0ooLzRBTF/gqQlACPac1lPsHI589dMRjhsCso9Mi8dBC8oI
2pvpETLx/JGGcvI9dUageKdK0rk37wOgsi0VEKHTlDpYWOwWGmEf3q5sa442Y0I4
a9fjB+YHp9HlKC6ZTdiNqfpcDebdIYPCKOzpdiVVLSNQGpwqCXBXIFu0dV/Ak3qX
S7DKgZIurAlxjQ3HSWUJsd0AVCV9K1iWVHpuorF4FKVWHodFDq7+/H6ctomAgunP
jvnqrj4tAgMBAAECggEAAv3dDJZVTjDLLhddgwwVfcGJ6APw34OR0Stzncf27uyl
uL2US+zcDGfuhERsFOFU4+RtqlDeRX55ARTaN/XS1R/UHIZiNtH0n+S+pbAyy/FK
HA/izUo+t7rMmdaun5pTN66SKdvpcrEHZyYVmcVu831LsczDAFue9xKYYXPDJQLu
Ea82zWt6TYT+N4QdabrU0o/kQI701tXj/adcitkync0qyT9lOZE6lGaRfbHred85
lcxiAN3FlqYtreoyhjDBDklJaql54wyr7CdgQXpOmePVMnHi7Kx6HhwVn/yrKxdE
fcutnj3mh0JHMazMGGxM7X+r4RXhNTggC3nA4LStgQKBgQD9pUXaQJgNwpmQ7l1B
ZpIhlYE89mA7YrdU9cdFTIgXTMklay5xgpSd7ulKwX0ZNkTDEDCc2RoB9ypE4jHD
aj1bxKlJnuvXwnmgHDnpsrhtJsFXn0qKe1irBJ3VACS3OFf7pP+4Obqcymm5ndjq
M/cZepuV6cqbNhm1cSmtNUSybQKBgQDgfEjlmGtMxT2ImDUwd2lNzT5HEz4s/P0l
4NDy8cPGTuYNyjjpY0LXslAJaRNwonKea2idiDiTWEVHPeG/zmHczQd+Ig5hpoYv
n24Cq5aW9hNTyq9PleEa2QWWbZYMCJVV9FTInnd3hKFZzOTTvu9nsKNN1WfaOS4H
U0X6Pi5iwQKBgFul3BeAP3C5X8N+XTPEXAjGfGwKmbrbcGLCa74eaQ4CMKvUjnN1
Oz3VlXXtc8YoVbAlqWsDBuKu7Bb3pAN337PI22I+ifjrzAaOLF7EtN67oiCG7egb
qW4hvOW5p4qMUT4b4Eowkb0VZh4rarU1EZOjOZRxZUOvyJpGyUhMdwYBAoGAfNii
Mg5ynl+TuUPtUOcYSYy79gtdqOeKYmaFzpdmqgN3LnQo8qOhqRQiLxmhFiNCW5ig
tfvsewW8gcKIqoO9KW9dm2iVVvml5xZjuFh1h7+TQCaZGUnhx2yrDt4jdM3RP9yC
ypBXIMFCew7YtGqb+q7iI4dsGpFyZ+CIKoQqiAECgYAFExiIjDSnxa+AYk6+lKT2
nhbWHW07WJvhlXeHDfGhjr8fY2BhtSYkvfU2t3NiWAbJCTACxZ7OcnKqYBHDcD0j
IDsZknvLXwwjcS0BVDtzRAgGBozdwE0tpet6loAwP0//q5ll/TguyEGCJGZhcxnX
sVbxA5EVF+OyBIjo6FzRRQ==
-----END PRIVATE KEY-----"""
    token = pyjwt.encode(payload, wrong_key, algorithm="RS256")
    response = rs256_client.get("/catalogs", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()


def test_rs256_rejects_malformed_token(rs256_client):
    response = rs256_client.get("/catalogs", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


def test_rs256_health_is_public_without_token(rs256_client):
    response = rs256_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}