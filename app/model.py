from typing import Iterable

import torch
import torch.nn.functional as f
from transformers import AutoModel, AutoTokenizer

from app.config import MAX_LEN, MODEL_DIR


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_encoder = None


def get_model():
    global _tokenizer, _encoder

    if _tokenizer is None or _encoder is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _encoder = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
        _encoder.eval()

    return _tokenizer, _encoder


def encode_texts(texts: Iterable[str]) -> torch.Tensor:
    tokenizer, encoder = get_model()
    batch_texts = [str(text) for text in texts]

    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {key: value.to(DEVICE) for key, value in enc.items()}

    with torch.no_grad():
        out = encoder(**enc)
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)

        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean = summed / counts

        return f.normalize(mean, p=2, dim=1)

