import os
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "biencoder_model"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "index_data"))
PRICE_LIST_DIR = Path(os.getenv("PRICE_LIST_DIR", BASE_DIR / "price_lists"))

MAX_LEN = int(os.getenv("MAX_LEN", "128"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
API_PREFIX = os.getenv("API_PREFIX", "").rstrip("/")
DISABLE_REQUEST_GUARD = os.getenv("DISABLE_REQUEST_GUARD", "false").lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str) -> list[str]:
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


ALLOWED_ORIGINS = _parse_csv(os.getenv("ALLOWED_ORIGINS", "https://api.turbo-metallurg-montazh.ru"))
ALLOWED_REFERER_ORIGINS = set(ALLOWED_ORIGINS)
ALLOWED_HOSTS = _parse_csv(os.getenv("ALLOWED_HOSTS", "api.turbo-metallurg-montazh.ru,emk-search-service,emk-search-service.backend.svc,localhost,127.0.0.1,testserver"))


def origin_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
