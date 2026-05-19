import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

from app.config import PRICE_LIST_DIR


ALLOWED_CATALOG_SUFFIXES = {".xls", ".xlsx"}


def parse_price(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"[^\d.,]", "", text)
    if not text:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".") if re.search(r",\d{1,2}$", text) else text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def sanitize_catalog_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace(" ", "_")
    if not name:
        raise ValueError("Empty filename")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_CATALOG_SUFFIXES:
        raise ValueError("Only .xls and .xlsx catalogs are supported")

    safe = re.sub(r"[^A-Za-zА-Яа-я0-9._-]", "_", name)
    if safe in {"", ".", ".."}:
        raise ValueError("Invalid filename")
    return safe


async def save_catalog_upload(file: UploadFile) -> dict[str, Any]:
    safe_name = sanitize_catalog_filename(file.filename or "")
    PRICE_LIST_DIR.mkdir(parents=True, exist_ok=True)
    target = PRICE_LIST_DIR / safe_name

    content = await file.read()
    if not content:
        raise ValueError("Uploaded catalog is empty")

    target.write_bytes(content)
    return {"filename": safe_name, "bytes": len(content), "path": str(target)}


def list_catalog_files() -> list[dict[str, Any]]:
    PRICE_LIST_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(PRICE_LIST_DIR.glob("*.xls*")):
        if path.suffix.lower() not in ALLOWED_CATALOG_SUFFIXES:
            continue
        stat = path.stat()
        files.append({"filename": path.name, "bytes": stat.st_size, "modified": stat.st_mtime})
    return files


def read_catalog_rows(path: Path, name_col: int = 0, price_col: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xls = pd.ExcelFile(path)

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, dtype=object)
        if df.shape[1] <= name_col:
            continue

        for idx, record in df.iterrows():
            raw_name = record.iloc[name_col]
            if pd.isna(raw_name):
                continue

            product_name = str(raw_name).strip()
            if not product_name or product_name.lower() == "nan":
                continue

            raw_price = record.iloc[price_col] if df.shape[1] > price_col else None
            rows.append(
                {
                    "name": product_name,
                    "price": parse_price(raw_price),
                    "source_file": path.name,
                    "sheet": sheet,
                    "row": int(idx) + 2,
                }
            )

    return rows


def load_price_lists() -> list[dict[str, Any]]:
    PRICE_LIST_DIR.mkdir(parents=True, exist_ok=True)
    catalog_paths = [
        path
        for path in sorted(PRICE_LIST_DIR.glob("*.xls*"))
        if path.suffix.lower() in ALLOWED_CATALOG_SUFFIXES
    ]
    if not catalog_paths:
        raise RuntimeError(f"No Excel files found in {PRICE_LIST_DIR}")

    best_by_name: dict[str, dict[str, Any]] = {}
    for path in catalog_paths:
        for row in read_catalog_rows(path):
            existing = best_by_name.get(row["name"])
            if existing is None:
                best_by_name[row["name"]] = row
                continue

            old_price = existing.get("price")
            new_price = row.get("price")
            if old_price is None and new_price is not None:
                best_by_name[row["name"]] = row
            elif old_price is not None and new_price is not None and new_price < old_price:
                best_by_name[row["name"]] = row

    rows = list(best_by_name.values())
    if not rows:
        raise RuntimeError("No valid product names found in uploaded catalogs")

    return rows

