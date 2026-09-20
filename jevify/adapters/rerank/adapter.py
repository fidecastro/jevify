"""RerankBackend: the backend port over a rerank endpoint (ADR-0002 D2, rerank kind).

Each option is scored as a document against the state as the query. Scores are
independent relevance judgments, so the answer is labeled `relevance`; the
engine's softmax over them is a ranking convenience, never a probability.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import time
from collections.abc import Callable, Sequence

from jevify.adapters.endpoint.client import OpenAICompatibleClient
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Question, State
from jevify.ports.backend import (
    BackendError,
    Capabilities,
    Dialect,
    RawAnswer,
    Rung,
    StateHandle,
)
from jevify.recipes.render import RenderedPrefix
from jevify.recipes.schema import Recipe

Clock = Callable[[], float]
_FLOOR = 1e-9


def _fill(template: str, **fields: str) -> str:
    for name, value in fields.items():
        template = template.replace("{" + name + "}", value)
    return template


class RerankBackend:
    def __init__(
        self, client: OpenAICompatibleClient, recipe: Recipe, *, clock: Clock = time.perf_counter
    ) -> None:
        if recipe.rerank is None or recipe.endpoint is None:
            raise BackendError("a rerank backend needs rerank and endpoint sections")
        self.client = client
        self.recipe = recipe
        self.spec = recipe.rerank
        self.clock = clock

    async def probe(self) -> Capabilities:
        query = _fill(self.spec.query, state="probe")
        try:
            response = await self.client.post_v1(
                self.spec.path,
                {"model": self.recipe.model.name, "query": query, "documents": ["probe"]},
            )
            scores = self._scores(response, 1)
        except BackendError as exc:
            raise BackendError(f"rerank endpoint did not answer the probe: {exc}") from exc
        notes = () if scores else ("rerank endpoint returned no scores for the probe",)
        return Capabilities(
            kind="rerank",
            model=self.recipe.model.name,
            dialect=Dialect(self.recipe.endpoint.dialect)
            if self.recipe.endpoint and self.recipe.endpoint.dialect != "auto"
            else None,
            modalities=frozenset({"text"}),
            rungs=frozenset(),
            token_ids_verified=False,
            probed_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            notes=notes,
        )

    async def warm(self, state: State) -> StateHandle:
        if state.modalities - {"text"}:
            raise BackendError("the rerank kind scores text only; the state carries image parts")
        query = _fill(self.spec.query, state=state.text)
        prefix = RenderedPrefix(mode="rerank", text=query)
        state_id = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return StateHandle(state_id=state_id, prefix=prefix)

    async def evaluate(self, handle: StateHandle, questions: Sequence[Question]) -> list[RawAnswer]:
        return [await self._answer(handle, q) for q in questions]

    def _documents(self, question: Question) -> tuple[list[str], list[str]]:
        if isinstance(question, ChoiceQuestion):
            labels = list(question.labels)
            documents = [
                _fill(
                    self.spec.document,
                    key=option.key,
                    description=(self.spec.description_sep + option.description)
                    if option.description
                    else "",
                    text=option.description or option.key,
                )
                for option in question.options
            ]
            return labels, documents
        if isinstance(question, NoulQuestion):
            return ["true"], [question.instructions or ""]
        raise BackendError(
            "the rerank kind answers choice and noul questions only; a score question needs "
            "a readout-capable recipe"
        )

    def _scores(self, response: dict, count: int) -> dict[int, float]:
        results = response.get("results") or response.get("data") or []
        scores: dict[int, float] = {}
        for item in results:
            index = item.get("index")
            value = item.get(self.spec.score_field, item.get("score"))
            if index is not None and value is not None:
                scores[int(index)] = float(value)
        return scores

    async def _answer(self, handle: StateHandle, question: Question) -> RawAnswer:
        labels, documents = self._documents(question)
        body = {
            "model": self.recipe.model.name,
            "query": handle.prefix.text,
            "documents": documents,
            "top_n": len(documents),
        }
        if self.spec.instruction:
            body["instruction"] = self.spec.instruction
        started = self.clock()
        response = await self.client.post_v1(self.spec.path, body)
        latency_ms = (self.clock() - started) * 1000.0
        scores = self._scores(response, len(documents))
        if len(scores) != len(documents):
            raise BackendError(
                f"rerank endpoint scored {len(scores)} of {len(documents)} documents"
            )
        by_label = {label: scores[i] for i, label in enumerate(labels)}
        if isinstance(question, NoulQuestion):
            s = min(1.0, max(0.0, by_label["true"]))
            logprobs = {"false": math.log(max(1.0 - s, _FLOOR)), "true": math.log(max(s, _FLOOR))}
        else:
            logprobs = {label: math.log(max(score, _FLOOR)) for label, score in by_label.items()}
        usage = response.get("usage") or {}
        return RawAnswer(
            question_id=question.id,
            logprobs=logprobs,
            semantics="relevance",
            rung=Rung.NATIVE,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            calls=1,
            composition="per_option",
            raw={"scores": by_label, "path": self.spec.path},
        )
