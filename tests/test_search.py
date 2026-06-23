import pytest
import torch

import app.search as search
import app.state as state


def test_find_similar_with_prices_requires_ready_index():
    with pytest.raises(RuntimeError, match="not built"):
        search.find_similar_with_prices("Автомат")


def test_find_similar_with_prices_returns_ranked_matches(monkeypatch):
    state.INDEX_READY = True
    state.CANDIDATES = [
        {"source_file": "a.xlsx", "sheet": "s1", "row": 2},
        {"source_file": "b.xlsx", "sheet": "s1", "row": 3},
    ]
    state.CANDIDATE_NAMES = ["Автомат", "Кабель"]
    state.CANDIDATE_PRICES = [100.0, 50.0]
    state.EMB_T = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(search, "encode_texts", lambda texts: torch.tensor([[0.1, 0.9]]))

    results = search.find_similar_with_prices("кабель", top_k=7)

    assert results == [
        {
            "name": "Кабель",
            "price": 50.0,
            "score": pytest.approx(0.9),
            "source_file": "b.xlsx",
            "sheet": "s1",
            "row": 3,
        },
        {
            "name": "Автомат",
            "price": 100.0,
            "score": pytest.approx(0.1),
            "source_file": "a.xlsx",
            "sheet": "s1",
            "row": 2,
        },
    ]
