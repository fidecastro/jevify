"""EndpointBackend: the backend port over an OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Sequence

from jevify.adapters.endpoint.client import OpenAICompatibleClient
from jevify.adapters.endpoint.dialects import base_request, parse_response
from jevify.adapters.endpoint.readout import RUNGS
from jevify.domain.questions import Question, State
from jevify.ports.backend import (
    BackendError,
    Capabilities,
    Dialect,
    RawAnswer,
    Rung,
    StateHandle,
)
from jevify.recipes.render import render_prefix, render_question
from jevify.recipes.schema import Recipe

Clock = Callable[[], float]


class EndpointBackend:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        recipe: Recipe,
        *,
        clock: Clock = time.perf_counter,
        concurrency: int = 1,
    ) -> None:
        if recipe.endpoint is None:
            raise BackendError("an endpoint backend needs a recipe with an endpoint section")
        self.client = client
        self.recipe = recipe
        self.clock = clock
        self.concurrency = max(1, concurrency)
        self.dialect = (
            Dialect(recipe.endpoint.dialect)
            if recipe.endpoint.dialect != "auto"
            else Dialect.GENERIC
        )

    async def probe(self) -> Capabilities:
        raise BackendError("probe is not implemented yet; pin readout.rung in the recipe")

    async def warm(self, state: State) -> StateHandle:
        prefix = render_prefix(self.recipe, state)
        identity = json.dumps(
            [prefix.mode, prefix.system, prefix.text, prefix.kwargs, len(prefix.images)],
            sort_keys=True,
            ensure_ascii=False,
        )
        state_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return StateHandle(state_id=state_id, prefix=prefix)

    async def evaluate(self, handle: StateHandle, questions: Sequence[Question]) -> list[RawAnswer]:
        rung = self._resolve_rung()
        gate = asyncio.Semaphore(self.concurrency)

        async def one(question: Question) -> RawAnswer:
            async with gate:
                return await self._answer(handle, question, rung)

        return list(await asyncio.gather(*(one(q) for q in questions)))

    def _resolve_rung(self) -> Rung:
        pinned = self.recipe.readout.rung
        if pinned == "auto":
            raise BackendError(
                "readout.rung is auto but the recipe has no probe section; "
                "run `jevify probe <recipe>` or pin a rung"
            )
        rung = Rung(pinned)
        if rung not in RUNGS:
            raise BackendError(f"rung {rung} is not implemented yet")
        return rung

    async def _answer(self, handle: StateHandle, question: Question, rung: Rung) -> RawAnswer:
        rendered = render_question(self.recipe, question)
        path, body = base_request(self.recipe.model.name, handle.prefix, rendered, self.dialect)
        impl = RUNGS[rung]
        impl.shape(body, rendered, self.recipe.readout.top_k, self.dialect)
        started = self.clock()
        response = await self.client.post_v1(path, body)
        latency_ms = (self.clock() - started) * 1000.0
        parsed = parse_response(path, response, self.dialect)
        readout = impl.read(parsed, rendered)
        return RawAnswer(
            question_id=question.id,
            logprobs=readout.logprobs,
            semantics="readout",
            rung=rung,
            degraded=readout.degraded,
            off_menu_mass=readout.off_menu_mass,
            latency_ms=latency_ms,
            prompt_tokens=parsed.prompt_tokens,
            cached_tokens=parsed.cached_tokens,
            missing=readout.missing,
            raw={"path": path, "entries": [e.__dict__ for e in parsed.entries]},
        )
