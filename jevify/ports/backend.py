"""The one backend port (ADR-0002 D1) and the registries of rungs and dialects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

from jevify.domain.questions import Question, Semantics, State
from jevify.recipes.render import RenderedPrefix


class Rung(StrEnum):
    """The readout ladder (ADR-0002 D4), strongest first."""

    NAMED_TOKEN_LOGPROBS = "named_token_logprobs"
    GRAMMAR = "grammar"
    TOP_K = "top_k"
    EQUAL_BIAS = "equal_bias"
    TOP_K_FLOOR = "top_k_floor"


RUNG_RANK: tuple[Rung, ...] = (
    Rung.NAMED_TOKEN_LOGPROBS,
    Rung.GRAMMAR,
    Rung.TOP_K,
    Rung.EQUAL_BIAS,
    Rung.TOP_K_FLOOR,
)


class Dialect(StrEnum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    GENERIC = "generic"


BackendKind = Literal["endpoint", "encoder", "rerank", "embedding"]


@dataclass(frozen=True)
class CacheEvidence:
    block_tokens: int | None
    first_ms: float | None
    second_ms: float | None
    cached_tokens: int | None


@dataclass(frozen=True)
class FanoutEvidence:
    concurrency: int
    concurrent_ms: float
    sequential_ms: float


@dataclass(frozen=True)
class Capabilities:
    """What the probe measured about a backend and its model."""

    kind: BackendKind
    model: str
    dialect: Dialect | None = None
    backend_version: str | None = None
    max_context: int | None = None
    modalities: frozenset[str] = frozenset({"text"})
    rungs: frozenset[Rung] = frozenset()
    top_k_cap: int | None = None
    logprobs_post_bias: bool | None = None
    prefill_honored: bool | None = None
    template_kwargs_honored: bool | None = None
    token_ids_verified: bool = False
    answer_tokens: dict[str, int] = field(default_factory=dict)
    multi_token_answers: tuple[str, ...] = ()
    cache: CacheEvidence | None = None
    fanout: FanoutEvidence | None = None
    slots: int | None = None  # llama.cpp parallel slots, each with its own cache
    probed_at: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """A plain, YAML-friendly record for the recipe's `probe` section."""
        payload = asdict(self)
        payload["dialect"] = str(self.dialect) if self.dialect else None
        payload["modalities"] = sorted(self.modalities)
        payload["rungs"] = [str(r) for r in RUNG_RANK if r in self.rungs]
        payload["multi_token_answers"] = list(self.multi_token_answers)
        payload["notes"] = list(self.notes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Capabilities:
        data = dict(payload)
        data["dialect"] = Dialect(data["dialect"]) if data.get("dialect") else None
        data["modalities"] = frozenset(data.get("modalities") or ["text"])
        data["rungs"] = frozenset(Rung(r) for r in data.get("rungs") or [])
        data["multi_token_answers"] = tuple(data.get("multi_token_answers") or ())
        data["notes"] = tuple(data.get("notes") or ())
        if data.get("cache") is not None:
            data["cache"] = CacheEvidence(**data["cache"])
        if data.get("fanout") is not None:
            data["fanout"] = FanoutEvidence(**data["fanout"])
        return cls(**data)


@dataclass(frozen=True)
class StateHandle:
    """A warmed state: the rendered prefix every question about it shares."""

    state_id: str
    prefix: RenderedPrefix
    warm_ms: float | None = None
    cached_tokens: int | None = None
    slots: tuple[int, ...] = ()  # llama.cpp slots that hold this prefix


@dataclass(frozen=True)
class RawAnswer:
    """One question's readout before the engine turns it into an Answer."""

    question_id: str
    logprobs: dict[str, float]
    semantics: Semantics
    rung: Rung
    degraded: bool = False
    off_menu_mass: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    missing: tuple[str, ...] = ()
    calls: int = 1
    composition: str = "single"
    raw: dict[str, Any] = field(default_factory=dict)


class BackendError(Exception):
    """A backend could not answer. Transport and HTTP failures are wrapped into this."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ContextLimitError(BackendError):
    """The input exceeded the backend's context; the message names the limit (INV-9)."""


class Backend(Protocol):
    async def probe(self) -> Capabilities: ...

    async def warm(self, state: State) -> StateHandle: ...

    async def evaluate(
        self, handle: StateHandle, questions: Sequence[Question]
    ) -> list[RawAnswer]: ...
