from __future__ import annotations

import json

import torch

import app.indexer as indexer
import app.state as state


def test_build_search_index_encodes_candidates_and_persists_files(tmp_path, monkeypatch):
    candidates = [
        {"name": "Автомат", "price": 100.0, "source_file": "a.xlsx"},
        {"name": "Кабель", "price": None, "source_file": "b.xlsx"},
    ]
    monkeypatch.setattr(indexer, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(indexer, "load_price_lists", lambda: candidates)

    def encode_texts(texts):
        if texts == ["Автомат"]:
            return torch.tensor([[1.0, 0.0]])
        if texts == ["Кабель"]:
            return torch.tensor([[0.0, 1.0]])
        raise AssertionError(texts)

    monkeypatch.setattr(indexer, "encode_texts", encode_texts)

    meta = indexer.build_search_index(batch_size=1)

    assert meta == {"items": 2, "dim": 2}
    assert state.INDEX_READY is True
    assert state.INDEX_STALE is False
    assert state.CANDIDATE_NAMES == ["Автомат", "Кабель"]
    assert state.CANDIDATE_PRICES == [100.0, None]
    assert torch.equal(state.EMB, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8")) == candidates
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8")) == meta


def test_load_index_from_disk_returns_false_when_files_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "INDEX_DIR", tmp_path)

    assert indexer.load_index_from_disk() is False
    assert state.INDEX_READY is False


def test_load_index_from_disk_restores_current_format(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "INDEX_DIR", tmp_path)
    candidates = [{"name": "Автомат", "price": 10.0}]
    torch.save(torch.tensor([[1.0, 0.0]]), tmp_path / "embeddings.pt")
    (tmp_path / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "names.json").write_text(json.dumps(["Автомат"], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "prices.json").write_text(json.dumps([10.0]), encoding="utf-8")

    assert indexer.load_index_from_disk() is True

    assert state.INDEX_READY is True
    assert state.INDEX_STALE is False
    assert state.CANDIDATES == candidates
    assert state.CANDIDATE_NAMES == ["Автомат"]
    assert state.CANDIDATE_PRICES == [10.0]
    assert torch.equal(state.EMB_T, torch.tensor([[1.0], [0.0]]))


def test_load_index_from_disk_supports_legacy_without_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "INDEX_DIR", tmp_path)
    torch.save(torch.tensor([[1.0]]), tmp_path / "embeddings.pt")
    (tmp_path / "names.json").write_text(json.dumps(["Автомат"], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "prices.json").write_text(json.dumps([10.0]), encoding="utf-8")

    assert indexer.load_index_from_disk() is True
    assert state.CANDIDATES == [{"name": "Автомат", "price": 10.0}]
