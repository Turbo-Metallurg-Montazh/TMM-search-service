from __future__ import annotations

import pytest

import app.state as state


@pytest.fixture(autouse=True)
def reset_search_state():
    state.reset_index()
    yield
    state.reset_index()
