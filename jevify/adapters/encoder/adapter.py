"""EncoderBackend: the backend port over an in-process encoder classifier (ADR-0002 D2,
encoder kind).

The `nli` layout: the state is the premise; each option, level or statement is a
hypothesis; the head's entailment and contradiction logits at each pair are the readout.
A choice or score answer carries, per label, the entailment-minus-contradiction logit (the
engine's softmax across labels ranks them, the same per-option composition the endpoint
kind uses for a yes/no reranker); a noul answer is the two-way distribution at the one
pair. The scores are the head's own outputs at the answer position, so the label is
`readout`. One forward pass scores every pair of a question.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from jevify.domain.questions import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    State,
)
from jevify.ports.backend import BackendError, Capabilities, RawAnswer, Rung, StateHandle
from jevify.recipes.render import RenderedPrefix
from jevify.recipes.schema import EncoderSpec, Recipe

Clock = Callable[[], float]
Pair = tuple[str, str]


def _fill(template: str, **fields: str) -> str:
    for name, value in fields.items():
        template = template.replace("{" + name + "}", value)
    return template


class Scorer(Protocol):
    """Premise/hypothesis pairs in, the head's logits by lower-cased label name out."""

    async def score(self, pairs: Sequence[Pair]) -> list[dict[str, float]]: ...

    def describe(self) -> dict[str, str | None]: ...


class EncoderBackend:
    def __init__(self, scorer: Scorer, recipe: Recipe, *, clock: Clock = time.perf_counter) -> None:
        if recipe.encoder is None:
            raise BackendError("an encoder backend needs an encoder section")
        self.scorer = scorer
        self.recipe = recipe
        self.spec: EncoderSpec = recipe.encoder
        self.clock = clock

    async def probe(self) -> Capabilities:
        try:
            [logits] = await self.scorer.score([("probe", "probe")])
        except BackendError as exc:
            raise BackendError(f"encoder model did not answer the probe: {exc}") from exc
        wanted = {self.spec.entailment_label.lower(), self.spec.contradiction_label.lower()}
        if not wanted <= set(logits):
            raise BackendError(
                f"the head's labels {sorted(logits)} lack the recipe's "
                f"{sorted(wanted)}; set encoder.entailment_label and encoder.contradiction_label"
            )
        described = self.scorer.describe()
        return Capabilities(
            kind="encoder",
            model=self.recipe.model.name,
            dialect=None,
            modalities=frozenset({"text"}),
            rungs=frozenset({Rung.NATIVE}),
            token_ids_verified=False,
            probed_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            notes=(
                f"layout {self.spec.layout}; head labels {', '.join(sorted(logits))}",
                f"transport {described.get('transport')}",
                "text only; one forward pass per question",
            ),
        )

    async def warm(self, state: State) -> StateHandle:
        if state.modalities - {"text"}:
            raise BackendError("the encoder kind reads text only; the state carries image parts")
        premise = _fill(self.spec.premise, state=state.text)
        prefix = RenderedPrefix(mode="encoder", text=premise)
        return StateHandle(
            state_id=hashlib.sha256(premise.encode("utf-8")).hexdigest(), prefix=prefix
        )

    async def evaluate(self, handle: StateHandle, questions: Sequence[Question]) -> list[RawAnswer]:
        return [await self._answer(handle, q) for q in questions]

    def _hypotheses(self, question: Question) -> list[str]:
        instructions = question.instructions or ""
        if isinstance(question, ChoiceQuestion):
            return [
                _fill(
                    self.spec.hypothesis,
                    instructions=instructions,
                    key=option.key,
                    description=(self.spec.description_sep + option.description)
                    if option.description
                    else "",
                    text=option.description or option.key,
                )
                for option in question.options
            ]
        if isinstance(question, ScoreQuestion):
            return [
                _fill(
                    self.spec.hypothesis,
                    instructions=instructions,
                    key=level,
                    description="",
                    text=level,
                )
                for level in question.levels
            ]
        assert isinstance(question, NoulQuestion)
        return [_fill(self.spec.noul_hypothesis, instructions=instructions)]

    async def _answer(self, handle: StateHandle, question: Question) -> RawAnswer:
        hypotheses = self._hypotheses(question)
        started = self.clock()
        scored = await self.scorer.score([(handle.prefix.text, h) for h in hypotheses])
        latency_ms = (self.clock() - started) * 1000.0
        entail, contra = self.spec.entailment_label.lower(), self.spec.contradiction_label.lower()
        gaps = []
        for logits in scored:
            if entail not in logits or contra not in logits:
                raise BackendError(
                    f"the head returned labels {sorted(logits)}, not {entail!r}/{contra!r}"
                )
            gaps.append(logits[entail] - logits[contra])
        if isinstance(question, NoulQuestion):
            [gap] = gaps
            p_true = 1.0 / (1.0 + math.exp(-gap))
            logprobs = {
                "false": math.log(max(1.0 - p_true, 1e-12)),
                "true": math.log(max(p_true, 1e-12)),
            }
            keys: tuple[str, ...] = ("true",)
        else:
            keys = question.labels
            logprobs = dict(zip(keys, gaps, strict=True))
        return RawAnswer(
            question_id=question.id,
            logprobs=logprobs,
            semantics="readout",
            rung=Rung.NATIVE,
            latency_ms=latency_ms,
            calls=1,
            composition="per_option",
            raw={
                "layout": self.spec.layout,
                "hypotheses": dict(zip(keys, hypotheses, strict=True)),
                "logits": dict(zip(keys, scored, strict=True)),
            },
        )
