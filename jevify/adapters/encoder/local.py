"""NliScorer: a sequence-classification head run in-process behind the `encoder` extra.

torch and transformers are imported inside methods so that importing this module never
needs the extra; a missing extra fails at first use with the install command named.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jevify.ports.backend import BackendError
from jevify.recipes.schema import LocalModelSpec

_BATCH = 16


class NliScorer:
    def __init__(self, spec: LocalModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._labels: list[str] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise BackendError(
                "the encoder kind needs the encoder extra: "
                "uv tool install 'jevify[encoder]' (or pip install 'jevify[encoder]')"
            ) from exc
        name = self.spec.device
        if name == "auto":
            name = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self._device = torch.device(name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.path)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.spec.path)
            .to(self._device)
            .eval()
            .requires_grad_(False)
        )
        id2label = self._model.config.id2label
        self._labels = [str(id2label[i]).lower() for i in range(len(id2label))]

    async def score(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        self._load()
        import torch

        out: list[dict[str, float]] = []
        for start in range(0, len(pairs), _BATCH):
            batch = list(pairs[start : start + _BATCH])
            encoded = self._tokenizer(
                [p for p, _ in batch], [h for _, h in batch], padding=True, return_tensors="pt"
            )
            if encoded.input_ids.shape[1] > self.spec.max_tokens:
                raise BackendError(
                    f"input exceeds the recipe's local.max_tokens={self.spec.max_tokens} "
                    f"({encoded.input_ids.shape[1]} tokens)"
                )
            with torch.no_grad():
                logits = self._model(**encoded.to(self._device)).logits.float().cpu()
            for row in logits.tolist():
                out.append(dict(zip(self._labels, row, strict=True)))
        return out

    def describe(self) -> dict[str, str | None]:
        return {"transport": "local", "dialect": None}
