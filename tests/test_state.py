import torch

import app.state as state


def test_reset_index_clears_all_runtime_state():
    state.INDEX_READY = True
    state.INDEX_STALE = True
    state.CANDIDATES = [{"name": "A"}]
    state.CANDIDATE_NAMES = ["A"]
    state.CANDIDATE_PRICES = [10.0]
    state.EMB = torch.tensor([[1.0]])
    state.EMB_T = torch.tensor([[1.0]])

    state.reset_index()

    assert state.INDEX_READY is False
    assert state.INDEX_STALE is False
    assert state.CANDIDATES == []
    assert state.CANDIDATE_NAMES == []
    assert state.CANDIDATE_PRICES == []
    assert state.EMB is None
    assert state.EMB_T is None


def test_mark_index_stale_sets_stale_flag_only():
    state.INDEX_READY = True

    state.mark_index_stale()

    assert state.INDEX_READY is True
    assert state.INDEX_STALE is True
