from typing import Any

import torch

import app.state as state
from app.model import encode_texts


def find_similar_with_prices(query: str, top_k: int = 7) -> list[dict[str, Any]]:
    if not state.INDEX_READY or state.EMB_T is None:
        raise RuntimeError("Search index not built")

    q_emb = encode_texts([query]).cpu()
    scores = torch.matmul(q_emb, state.EMB_T).squeeze(0)

    k = min(top_k, scores.shape[0])
    top = torch.topk(scores, k=k)

    results = []
    for idx, score in zip(top.indices.tolist(), top.values.tolist(), strict=False):
        candidate = state.CANDIDATES[idx] if idx < len(state.CANDIDATES) else {}
        results.append(
            {
                "name": state.CANDIDATE_NAMES[idx],
                "price": state.CANDIDATE_PRICES[idx],
                "score": float(score),
                "source_file": candidate.get("source_file"),
                "sheet": candidate.get("sheet"),
                "row": candidate.get("row"),
            }
        )

    return results

