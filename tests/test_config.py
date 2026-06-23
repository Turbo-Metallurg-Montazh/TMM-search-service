import importlib

import app.config as config
from app.config import _parse_csv, origin_from_url


def test_parse_csv_trims_values_and_trailing_slashes():
    assert _parse_csv(" https://example.com/ , localhost ,, ") == [
        "https://example.com",
        "localhost",
    ]


def test_origin_from_url_returns_scheme_and_host():
    assert origin_from_url("https://example.com/path?q=1") == "https://example.com"
    assert origin_from_url("http://localhost:8000/docs") == "http://localhost:8000"


def test_origin_from_url_rejects_invalid_values():
    assert origin_from_url("not-a-url") is None
    assert origin_from_url("//example.com/no-scheme") is None


def test_disable_request_guard_env_parsing(monkeypatch):
    monkeypatch.setenv("DISABLE_REQUEST_GUARD", "true")
    reloaded = importlib.reload(config)

    assert reloaded.DISABLE_REQUEST_GUARD is True

    monkeypatch.delenv("DISABLE_REQUEST_GUARD")
    importlib.reload(config)
