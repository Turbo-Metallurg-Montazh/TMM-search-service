from typing import Any

import torch


INDEX_READY: bool = False
INDEX_STALE: bool = False

CANDIDATES: list[dict[str, Any]] = []
CANDIDATE_NAMES: list[str] = []
CANDIDATE_PRICES: list[float | None] = []

EMB: torch.Tensor | None = None
EMB_T: torch.Tensor | None = None


def reset_index() -> None:
    global INDEX_READY, INDEX_STALE, CANDIDATES, CANDIDATE_NAMES, CANDIDATE_PRICES, EMB, EMB_T

    INDEX_READY = False
    INDEX_STALE = False
    CANDIDATES = []
    CANDIDATE_NAMES = []
    CANDIDATE_PRICES = []
    EMB = None
    EMB_T = None


def mark_index_stale() -> None:
    global INDEX_STALE

    INDEX_STALE = True

