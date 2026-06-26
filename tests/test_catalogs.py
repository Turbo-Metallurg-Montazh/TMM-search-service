from __future__ import annotations

import asyncio
from io import BytesIO

import pandas as pd
import pytest
from fastapi import UploadFile

import app.catalogs as catalogs


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("нет цены", None),
        (123, 123.0),
        (123.45, 123.45),
        ("1,234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("1,234", 1234.0),
    ],
)
def test_parse_price(raw, expected):
    assert catalogs.parse_price(raw) == expected


def test_parse_price_with_currency_text_and_trailing_punctuation():
    assert catalogs.parse_price("1 234,56 руб.") == 1234.56


def test_parse_price_ignores_bool():
    assert catalogs.parse_price(True) is None


def test_sanitize_catalog_filename_accepts_safe_excel_names():
    assert catalogs.sanitize_catalog_filename("../Прайс лист.xlsx") == "Прайс_лист.xlsx"
    assert catalogs.sanitize_catalog_filename("../Прайс лист.csv") == "Прайс_лист.csv"


@pytest.mark.parametrize("filename", ["", "catalog.txt", ".."])
def test_sanitize_catalog_filename_rejects_invalid_names(filename):
    with pytest.raises(ValueError):
        catalogs.sanitize_catalog_filename(filename)


def test_list_catalog_files_returns_excel_files_only(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "PRICE_LIST_DIR", tmp_path)
    (tmp_path / "a.xlsx").write_bytes(b"x")
    (tmp_path / "b.xls").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")

    files = catalogs.list_catalog_files()

    assert [item["filename"] for item in files] == ["a.xlsx", "b.xls"]
    assert all(item["bytes"] == 1 for item in files)


def test_read_catalog_rows_reads_all_sheets_and_prices(tmp_path):
    path = tmp_path / "catalog.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["Автомат 16А", "1 000,50"], [None, 1], ["", 2]], columns=["name", "price"]).to_excel(
            writer,
            index=False,
            sheet_name="main",
        )
        pd.DataFrame([["Кабель", 55]], columns=["name", "price"]).to_excel(
            writer,
            index=False,
            sheet_name="extra",
        )

    rows = catalogs.read_catalog_rows(path)

    assert rows == [
        {
            "name": "Автомат 16А",
            "price": 1000.50,
            "source_file": "catalog.xlsx",
            "sheet": "main",
            "row": 2,
        },
        {
            "name": "Кабель",
            "price": 55.0,
            "source_file": "catalog.xlsx",
            "sheet": "extra",
            "row": 2,
        },
    ]


def test_load_price_lists_keeps_cheapest_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "PRICE_LIST_DIR", tmp_path)
    with pd.ExcelWriter(tmp_path / "catalog.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([["Автомат", 200], ["Автомат", 150], ["Реле", None]], columns=["name", "price"]).to_excel(
            writer,
            index=False,
        )

    rows = catalogs.load_price_lists()

    assert rows == [
        {
            "name": "Автомат",
            "price": 150.0,
            "source_file": "catalog.xlsx",
            "sheet": "Sheet1",
            "row": 3,
        },
        {
            "name": "Реле",
            "price": None,
            "source_file": "catalog.xlsx",
            "sheet": "Sheet1",
            "row": 4,
        },
    ]


def test_load_price_lists_rejects_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "PRICE_LIST_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="No Excel files"):
        catalogs.load_price_lists()


def test_save_catalog_upload_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "PRICE_LIST_DIR", tmp_path)
    upload = UploadFile(filename="Прайс.xlsx", file=BytesIO(b"content"))

    result = asyncio.run(catalogs.save_catalog_upload(upload))

    assert result == {
        "filename": "Прайс.xlsx",
        "bytes": 7,
        "path": str(tmp_path / "Прайс.xlsx"),
    }
    assert (tmp_path / "Прайс.xlsx").read_bytes() == b"content"


def test_save_catalog_upload_rejects_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "PRICE_LIST_DIR", tmp_path)
    upload = UploadFile(filename="empty.xlsx", file=BytesIO(b""))

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(catalogs.save_catalog_upload(upload))


def test_read_csv_with_comma_separator(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text(
        "Автомат 16А,1234.56\nКабель,100\n",
        encoding="utf-8",
    )
    rows = catalogs.read_catalog_rows(path)
    assert rows == [
        {
            "name": "Автомат 16А",
            "price": 1234.56,
            "source_file": "catalog.csv",
            "sheet": None,
            "row": 2,
        },
        {
            "name": "Кабель",
            "price": 100.0,
            "source_file": "catalog.csv",
            "sheet": None,
            "row": 3,
        },
    ]
@pytest.mark.parametrize(
    "separator",
    [
        ";",
        "\t",
    ],
)


def test_read_csv_with_different_separators(tmp_path, separator):
    path = tmp_path / "catalog.csv"
    path.write_text(
        f"Автомат{separator}1234.56\n",
        encoding="utf-8",
    )
    rows = catalogs.read_catalog_rows(path)
    assert rows[0]["name"] == "Автомат"
    assert rows[0]["price"] == 1234.56


def test_read_csv_supports_cyrillic(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text(
        "Выключатель;250\nРозетка;500\n",
        encoding="utf-8",
    )
    rows = catalogs.read_catalog_rows(path)
    assert [row["name"] for row in rows] == [
        "Выключатель",
        "Розетка",
    ]


def test_read_csv_skips_empty_rows(tmp_path):
    path = tmp_path / "catalog.csv"

    path.write_text(
        "Автомат,100\n\n,200\nКабель,300\n",
        encoding="utf-8",
    )
    rows = catalogs.read_catalog_rows(path)
    assert [row["name"] for row in rows] == [
        "Автомат",
        "Кабель",
    ]
@pytest.mark.parametrize(
    ("price", "expected"),
    [
        ("1234.56", 1234.56),
        ("1 234,56", 1234.56),
        ("1 234 руб.", 1234.0),
    ],
)


def test_read_csv_price_formats(tmp_path, price, expected):
    path = tmp_path / "catalog.csv"
    path.write_text(
        f'Товар,"{price}"\n',
        encoding="utf-8",
    )
    rows = catalogs.read_catalog_rows(path)
    assert rows[0]["price"] == expected


def test_load_price_lists_includes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(catalogs, "PRICE_LIST_DIR", tmp_path)
    (tmp_path / "catalog.csv").write_text(
        "Автомат,150\n",
        encoding="utf-8",
    )
    rows = catalogs.load_price_lists()
    assert rows == [
        {
            "name": "Автомат",
            "price": 150.0,
            "source_file": "catalog.csv",
            "sheet": None,
            "row": 2,
        }
    ]