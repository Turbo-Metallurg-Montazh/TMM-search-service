from __future__ import annotations

from types import SimpleNamespace

import torch

import app.model as model


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        self.texts = texts
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2], [3, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
        }


class FakeEncoder:
    def __init__(self):
        self.eval_called = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True

    def __call__(self, **kwargs):
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [
                    [[3.0, 0.0], [0.0, 4.0]],
                    [[10.0, 0.0], [99.0, 99.0]],
                ]
            )
        )


def test_get_model_loads_tokenizer_and_encoder_once(monkeypatch):
    tokenizer = FakeTokenizer()
    encoder = FakeEncoder()
    calls = {"tokenizer": 0, "encoder": 0}
    model._tokenizer = None
    model._encoder = None

    def load_tokenizer(path):
        calls["tokenizer"] += 1
        return tokenizer

    def load_encoder(path):
        calls["encoder"] += 1
        return encoder

    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(model.AutoModel, "from_pretrained", load_encoder)

    assert model.get_model() == (tokenizer, encoder)
    assert model.get_model() == (tokenizer, encoder)
    assert calls == {"tokenizer": 1, "encoder": 1}
    assert encoder.eval_called is True


def test_encode_texts_normalizes_masked_mean(monkeypatch):
    tokenizer = FakeTokenizer()
    encoder = FakeEncoder()
    monkeypatch.setattr(model, "get_model", lambda: (tokenizer, encoder))
    monkeypatch.setattr(model, "DEVICE", "cpu")

    embeddings = model.encode_texts(["a", 123])

    assert tokenizer.texts == ["a", "123"]
    assert tokenizer.kwargs["padding"] is True
    assert tokenizer.kwargs["truncation"] is True
    assert embeddings.shape == (2, 2)
    assert torch.allclose(embeddings[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(embeddings[1], torch.tensor([1.0, 0.0]))
