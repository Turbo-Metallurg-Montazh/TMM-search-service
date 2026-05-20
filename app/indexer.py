import json
from typing import Any

import torch

import app.state as state
from app.catalogs import load_price_lists
from app.config import DEFAULT_BATCH_SIZE, INDEX_DIR
from app.model import encode_texts


def _candidate_prices(candidates: list[dict[str, Any]]) -> list[float | None]:
    return [candidate.get("price") for candidate in candidates]


def _candidate_names(candidates: list[dict[str, Any]]) -> list[str]:
    return [str(candidate["name"]) for candidate in candidates]


def build_search_index(batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    candidates = load_price_lists()
    names = _candidate_names(candidates)

    embeddings = []
    for i in range(0, len(names), batch_size):
        embeddings.append(encode_texts(names[i : i + batch_size]).cpu())

    emb = torch.cat(embeddings, dim=0)

    state.CANDIDATES = candidates
    state.CANDIDATE_NAMES = names
    state.CANDIDATE_PRICES = _candidate_prices(candidates)
    state.EMB = emb
    state.EMB_T = emb.t()
    state.INDEX_READY = True
    state.INDEX_STALE = False

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(emb, INDEX_DIR / "embeddings.pt")
    (INDEX_DIR / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "names.json").write_text(json.dumps(state.CANDIDATE_NAMES, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "prices.json").write_text(json.dumps(state.CANDIDATE_PRICES), encoding="utf-8")

    meta = {"items": len(names), "dim": int(emb.shape[1])}
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def load_index_from_disk() -> bool:
    emb_path = INDEX_DIR / "embeddings.pt"
    candidates_path = INDEX_DIR / "candidates.json"
    names_path = INDEX_DIR / "names.json"
    prices_path = INDEX_DIR / "prices.json"

    if not (emb_path.exists() and names_path.exists() and prices_path.exists()):
        return False

    state.EMB = torch.load(emb_path, map_location="cpu", weights_only=False)
    state.EMB_T = state.EMB.t()
    state.CANDIDATE_NAMES = json.loads(names_path.read_text(encoding="utf-8"))
    state.CANDIDATE_PRICES = json.loads(prices_path.read_text(encoding="utf-8"))

    if candidates_path.exists():
        state.CANDIDATES = json.loads(candidates_path.read_text(encoding="utf-8"))
    else:
        state.CANDIDATES = [
            {"name": name, "price": price}
            for name, price in zip(state.CANDIDATE_NAMES, state.CANDIDATE_PRICES, strict=False)
        ]

    state.INDEX_READY = True
    state.INDEX_STALE = False
    return True
