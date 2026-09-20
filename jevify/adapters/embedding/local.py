"""LocalEmbedder: an in-process embedding model behind the `encoder` extra.

Ported from decision-poc's `Embedding`: left padding, last-token pooling (Qwen3-Embedding's
reference recipe) or mean pooling, float16 off CPU. torch and transformers are imported
inside methods so that importing this module never needs the extra; a missing extra fails
at first use with the install command named.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jevify.ports.backend import BackendError
from jevify.recipes.schema import LocalModelSpec

_BATCH = 16


class LocalEmbedder:
    def __init__(self, spec: LocalModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise BackendError(
                "the local embedding kind needs the encoder extra: "
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
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.path, padding_side="left")
        self._model = (
            AutoModel.from_pretrained(
                self.spec.path,
                dtype=torch.float32 if self._device.type == "cpu" else torch.float16,
            )
            .to(self._device)
            .eval()
            .requires_grad_(False)
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        import torch

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = list(texts[start : start + _BATCH])
            encoded = self._tokenizer(batch, padding=True, return_tensors="pt")
            if encoded.input_ids.shape[1] > self.spec.max_tokens:
                raise BackendError(
                    f"input exceeds the recipe's local.max_tokens={self.spec.max_tokens} "
                    f"({encoded.input_ids.shape[1]} tokens)"
                )
            encoded = encoded.to(self._device)
            with torch.no_grad():
                hidden = self._model(**encoded, use_cache=False).last_hidden_state.float()
            if self.spec.pooling == "last":
                pooled = hidden[:, -1]  # left padding puts every last real token at -1
            else:
                mask = encoded.attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            vectors.extend(pooled.cpu().tolist())
        return vectors

    def describe(self) -> dict[str, str | None]:
        return {"transport": "local", "dialect": None}
