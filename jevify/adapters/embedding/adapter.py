"""EmbeddingBackend: the backend port over an embedding model (ADR-0002 D2, embedding kind).

The state (with the question's instructions, for an instruction-tuned model) becomes one
query vector; each option becomes one document vector; the answer is the cosine between
them. Cosines are independent similarities, so the answer is labeled `similarity`; the
engine's softmax over cosine times the recipe's scale is a ranking convenience, never a
probability. Only choice questions are answerable: a noul or score question has no menu of
texts to compare against.

The embedder behind the backend is a narrow inner seam (`Embedder`): one implementation
posts to an OpenAI-compatible `/v1/embeddings` route, the other runs a model in-process
behind the `encoder` extra. Both cache option vectors, because a menu repeats across states
far more often than a state repeats across menus.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Protocol

from jevify.adapters.endpoint.client import OpenAICompatibleClient
from jevify.domain.questions import ChoiceQuestion, Question, State
from jevify.ports.backend import (
    BackendError,
    Capabilities,
    Dialect,
    RawAnswer,
    Rung,
    StateHandle,
)
from jevify.recipes.render import RenderedPrefix
from jevify.recipes.schema import EmbeddingSpec, Recipe

Clock = Callable[[], float]
Vector = list[float]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _fill(template: str, **fields: str) -> str:
    for name, value in fields.items():
        template = template.replace("{" + name + "}", value)
    return template


class Embedder(Protocol):
    """One batch of texts in, one vector per text out, in order."""

    async def embed(self, texts: Sequence[str]) -> list[Vector]: ...

    def describe(self) -> dict[str, str | None]: ...


class EndpointEmbedder:
    def __init__(self, client: OpenAICompatibleClient, recipe: Recipe) -> None:
        assert recipe.embedding is not None and recipe.endpoint is not None
        self.client = client
        self.recipe = recipe
        self.path = recipe.embedding.path

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        body = {
            "model": self.recipe.model.name,
            "input": texts[0] if len(texts) == 1 else list(texts),
        }
        response = await self.client.post_v1(self.path, body)
        data = sorted(response.get("data") or [], key=lambda item: item.get("index", 0))
        vectors = [list(map(float, item["embedding"])) for item in data]
        if len(vectors) != len(texts):
            raise BackendError(
                f"embedding endpoint returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def describe(self) -> dict[str, str | None]:
        assert self.recipe.endpoint is not None
        dialect = self.recipe.endpoint.dialect
        return {"transport": "endpoint", "dialect": None if dialect == "auto" else dialect}


class _Lru:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: OrderedDict[str, Vector] = OrderedDict()

    def get(self, key: str) -> Vector | None:
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        return None

    def put(self, key: str, value: Vector) -> None:
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)


class EmbeddingBackend:
    def __init__(
        self, embedder: Embedder, recipe: Recipe, *, clock: Clock = time.perf_counter
    ) -> None:
        if recipe.embedding is None:
            raise BackendError("an embedding backend needs an embedding section")
        self.embedder = embedder
        self.recipe = recipe
        self.spec: EmbeddingSpec = recipe.embedding
        self.clock = clock
        self.queries = _Lru(64)
        self.options = _Lru(4096)

    async def probe(self) -> Capabilities:
        try:
            [vector] = await self.embedder.embed(["probe"])
        except BackendError as exc:
            raise BackendError(f"embedding model did not answer the probe: {exc}") from exc
        described = self.embedder.describe()
        dialect = described.get("dialect")
        return Capabilities(
            kind="embedding",
            model=self.recipe.model.name,
            dialect=Dialect(dialect) if dialect else None,
            modalities=frozenset({"text"}),
            rungs=frozenset(),
            token_ids_verified=False,
            probed_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            notes=(
                f"embedding dimension {len(vector)}",
                f"transport {described.get('transport')}",
                "choice questions only; cosine similarity, labeled similarity",
            ),
        )

    async def warm(self, state: State) -> StateHandle:
        if state.modalities - {"text"}:
            raise BackendError(
                "the embedding kind compares text only; the state carries image parts"
            )
        prefix = RenderedPrefix(mode="embedding", text=state.text)
        state_id = hashlib.sha256(state.text.encode("utf-8")).hexdigest()
        return StateHandle(state_id=state_id, prefix=prefix)

    async def evaluate(self, handle: StateHandle, questions: Sequence[Question]) -> list[RawAnswer]:
        return [await self._answer(handle, q) for q in questions]

    def _option_texts(self, question: ChoiceQuestion) -> list[str]:
        return [
            _fill(
                self.spec.option,
                key=option.key,
                description=(self.spec.description_sep + option.description)
                if option.description
                else "",
                text=option.description or option.key,
            )
            for option in question.options
        ]

    async def _answer(self, handle: StateHandle, question: Question) -> RawAnswer:
        if not isinstance(question, ChoiceQuestion):
            raise BackendError(
                "the embedding kind answers choice questions only; a noul or score question "
                "needs a readout-capable or rerank recipe"
            )
        started = self.clock()
        calls = 0
        query_text = _fill(
            self.spec.query, instructions=question.instructions or "", state=handle.prefix.text
        )
        query = self.queries.get(query_text)
        if query is None:
            [query] = await self.embedder.embed([query_text])
            self.queries.put(query_text, query)
            calls += 1
        texts = self._option_texts(question)
        missing = list(dict.fromkeys(t for t in texts if self.options.get(t) is None))
        if missing:
            for text, vector in zip(missing, await self.embedder.embed(missing), strict=True):
                self.options.put(text, vector)
            calls += 1
        cosines = {
            label: cosine(query, self.options.get(text) or [])
            for label, text in zip(question.labels, texts, strict=True)
        }
        return RawAnswer(
            question_id=question.id,
            logprobs={label: value * self.spec.scale for label, value in cosines.items()},
            semantics="similarity",
            rung=Rung.NATIVE,
            latency_ms=(self.clock() - started) * 1000.0,
            calls=calls,
            composition="per_option",
            raw={"cosine": cosines, "scale": self.spec.scale, "query_cached": calls == 0},
        )
