"""The Engine: warm a state, evaluate typed questions, return Answers with semantics.

The engine is the only caller of the backend port and the only producer of
`Answer`. It never renders prompts (the recipe does) and never reads logprobs
(the adapter does); it turns raw readouts into distributions, confidences and
usage, and it will own the permutation policy.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from jevify.domain.distribution import Distribution
from jevify.domain.questions import (
    Question,
    QuestionKind,
    ScoreQuestion,
    Semantics,
    State,
)
from jevify.ports.backend import Backend, RawAnswer, StateHandle
from jevify.recipes.schema import Recipe

Clock = Callable[[], float]


@dataclass(frozen=True)
class ReadoutRecord:
    rung: str
    degraded: bool
    off_menu_mass: float | None
    permutation: str
    calls: int
    latency_ms: float
    prompt_tokens: int | None
    cached_tokens: int | None
    missing: tuple[str, ...] = ()
    temperature: float | None = None


@dataclass(frozen=True)
class Answer:
    question_id: str
    kind: QuestionKind
    distribution: Distribution
    confidence: float
    semantics: Semantics
    readout: ReadoutRecord
    legend: dict[str, str] = field(default_factory=dict)

    @property
    def selected(self) -> str:
        return self.distribution.argmax()

    @property
    def expectation(self) -> float:
        return self.distribution.expectation()

    @property
    def probability_true(self) -> float:
        return self.distribution.as_mapping()["true"]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Evaluation:
    state_id: str
    answers: list[Answer]
    usage: Usage
    warm_ms: float | None
    elapsed_ms: float


class Engine:
    def __init__(
        self, backend: Backend, recipe: Recipe, *, clock: Clock = time.perf_counter
    ) -> None:
        self.backend = backend
        self.recipe = recipe
        self.clock = clock

    async def warm(self, state: State) -> StateHandle:
        return await self.backend.warm(state)

    async def evaluate(self, handle: StateHandle, questions: Sequence[Question]) -> Evaluation:
        started = self.clock()
        raw = await self.backend.evaluate(handle, questions)
        by_id = {q.id: q for q in questions}
        answers = [self._answer(by_id[r.question_id], r) for r in raw]
        usage = Usage(
            input_tokens=sum(r.prompt_tokens or 0 for r in raw),
            output_tokens=len(raw),
        )
        return Evaluation(
            state_id=handle.state_id,
            answers=answers,
            usage=usage,
            warm_ms=handle.warm_ms,
            elapsed_ms=(self.clock() - started) * 1000.0,
        )

    async def ask(self, state: State, questions: Sequence[Question]) -> Evaluation:
        handle = await self.warm(state)
        return await self.evaluate(handle, questions)

    def _answer(self, question: Question, raw: RawAnswer) -> Answer:
        ordered = {label: raw.logprobs[label] for label in question.labels if label in raw.logprobs}
        temperature = self._temperature(question, raw)
        if temperature is not None:
            ordered = {label: value / temperature for label, value in ordered.items()}
        distribution = Distribution.from_logprobs(ordered)
        legend = (
            {str(i): level for i, level in enumerate(question.levels)}
            if isinstance(question, ScoreQuestion)
            else {}
        )
        return Answer(
            question_id=question.id,
            kind=question.kind,
            distribution=distribution,
            confidence=distribution.confidence(question.kind),
            semantics="calibrated" if temperature is not None else raw.semantics,
            readout=ReadoutRecord(
                rung=str(raw.rung),
                degraded=raw.degraded,
                off_menu_mass=raw.off_menu_mass,
                permutation=self.recipe.readout.permutation,
                calls=raw.calls,
                latency_ms=raw.latency_ms,
                prompt_tokens=raw.prompt_tokens,
                cached_tokens=raw.cached_tokens,
                missing=raw.missing,
                temperature=temperature,
            ),
            legend=legend,
        )

    def _temperature(self, question: Question, raw: RawAnswer) -> float | None:
        """A fitted temperature applies only to readouts, only for a bucket the table covers."""
        table = self.recipe.calibration
        if table is None or raw.semantics != "readout":
            return None
        from jevify.evaluation.calibrate import bucket_key

        return table.temperatures.get(bucket_key(question.kind, len(question.labels)))
