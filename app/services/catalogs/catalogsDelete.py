from pathlib import Path
from app.services.catalogs.catalogs import ALLOWED_CATALOG_SUFFIXES
from pydantic import BaseModel

from app.config import PRICE_LIST_DIR

def validate_delete_catalog_request(filenames: list[str]) -> None:
    if not filenames:
        raise ValueError(f"Invalid filename")
    for filename in filenames:
        safe_name = Path(filename).name

        if safe_name != filename:
            raise ValueError(f"Invalid filename")
        if Path(filename).suffix.lower() not in ALLOWED_CATALOG_SUFFIXES:
            raise ValueError(f"Not supported file type")


def delete_catalog_files(filenames: list[str]) -> list[str]:
    PRICE_LIST_DIR.mkdir(parents=True, exist_ok=True)
    deleted = []
    for filename in filenames:
        target = (PRICE_LIST_DIR / filename).resolve()
        if not str(target).startswith(str(PRICE_LIST_DIR.resolve())):
            raise ValueError(f"Path traversal detected: {filename}")
        if not target.exists():
            raise FileNotFoundError(filename)

        target.unlink()
        deleted.append(filename)

    return deleted

class CatalogDeleteItem(BaseModel):
    filename: str
    extension: str


class DeleteCatalogsRequest(BaseModel):
    files: list[CatalogDeleteItem]