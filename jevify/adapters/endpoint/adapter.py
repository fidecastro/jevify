"""EndpointBackend: the backend port over an OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence

from jevify.adapters.endpoint.client import OpenAICompatibleClient
from jevify.adapters.endpoint.dialects import base_request, parse_response
from jevify.adapters.endpoint.readout import RUNGS, MissingLabelsError
from jevify.domain.distribution import Distribution
from jevify.domain.questions import Question, State
from jevify.ports.backend import (
    RUNG_RANK,
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
        self.dialect = (
            Dialect(recipe.endpoint.dialect)
            if recipe.endpoint.dialect != "auto"
            else Dialect.GENERIC
        )
        self.capabilities = Capabilities.from_dict(recipe.probe) if recipe.probe else None
        probed = (
            self.capabilities.fanout.concurrency
            if self.capabilities and self.capabilities.fanout
            else 1
        )
        self.concurrency = max(1, concurrency, probed)
        self.slots = self.capabilities.slots if self.capabilities else None

    async def probe(self) -> Capabilities:
        from jevify.adapters.endpoint.probe import Prober

        return await Prober(self.client, self.recipe, self.clock).run()

    async def warm(self, state: State) -> StateHandle:
        """Send the state-only prefix once (one token, nothing read) so the server's cache
        holds it; on llama.cpp once per slot the fan-out will use."""
        wanted = state.modalities - {"text"}
        if wanted and self.capabilities and not wanted <= self.capabilities.modalities:
            missing = sorted(wanted - self.capabilities.modalities)
            raise BackendError(
                f"this backend was probed without {missing} support; the state carries "
                f"{missing} parts and jevify never drops them silently (INV-9)"
            )
        prefix = render_prefix(self.recipe, state)
        identity = json.dumps(
            [prefix.mode, prefix.system, prefix.text, prefix.kwargs, len(prefix.images)],
            sort_keys=True,
            ensure_ascii=False,
        )
        state_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        slots = tuple(range(min(self.concurrency, self.slots))) if self.slots else ()
        started = self.clock()
        cached: int | None = None
        for slot in slots or (None,):
            path, body = base_request(self.recipe.model.name, prefix, "", self.dialect)
            if slot is not None:
                body["id_slot"] = slot
            post = self.client.post_root if path == "completion" else self.client.post_v1
            parsed = parse_response(path, await post(path, body), self.dialect)
            cached = parsed.cached_tokens
        warm_ms = (self.clock() - started) * 1000.0
        return StateHandle(
            state_id=state_id, prefix=prefix, warm_ms=warm_ms, cached_tokens=cached, slots=slots
        )

    async def evaluate(self, handle: StateHandle, questions: Sequence[Question]) -> list[RawAnswer]:
        ladder = self._ladder()
        gate = asyncio.Semaphore(self.concurrency)

        async def one(index: int, question: Question) -> RawAnswer:
            slot = handle.slots[index % len(handle.slots)] if handle.slots else None
            async with gate:
                for step, rung in enumerate(ladder):
                    try:
                        return await self._answer(handle, question, rung, slot)
                    except MissingLabelsError:
                        if step == len(ladder) - 1:
                            raise
                raise AssertionError("unreachable: the ladder is never empty")

        return list(await asyncio.gather(*(one(i, q) for i, q in enumerate(questions))))

    def _ladder(self) -> list[Rung]:
        """The rungs a question may be read on, strongest first. A pinned rung stands alone
        and fails loudly; auto walks every proven rung down to the floor."""
        strongest = self._resolve_rung()
        if self.recipe.readout.rung != "auto":
            return [strongest]
        proven = Capabilities.from_dict(self.recipe.probe).rungs if self.recipe.probe else set()
        return [r for r in RUNG_RANK if r in proven and r in RUNGS]

    def _resolve_rung(self) -> Rung:
        pinned = self.recipe.readout.rung
        proven = (
            Capabilities.from_dict(self.recipe.probe).rungs if self.recipe.probe else frozenset()
        )
        if pinned == "auto":
            if not self.recipe.probe:
                raise BackendError(
                    "readout.rung is auto but the recipe has no probe section; "
                    "run `jevify probe <recipe>` or pin a rung"
                )
            for rung in RUNG_RANK:
                if rung in proven and rung in RUNGS:
                    return rung
            raise BackendError("the probe proved no rung this build implements")
        rung = Rung(pinned)
        if rung not in RUNGS:
            raise BackendError(f"rung {rung} is not implemented yet")
        if self.recipe.probe and rung not in proven:
            raise BackendError(
                f"readout.rung {rung} is pinned but the probe did not prove it; "
                f"proven: {[str(r) for r in RUNG_RANK if r in proven]}"
            )
        return rung

    def _orders(self, question: Question, rendered) -> list[tuple[int, ...]] | None:
        """Cyclic option orders when the recipe asks for permutation on an identifier-based
        choice or score question; None means a single order."""
        policy = self.recipe.readout.permutation
        n = len(question.labels)
        if policy == "none" or rendered.composition != "single" or n < 2:
            return None
        if policy == "full":
            import itertools

            return list(itertools.permutations(range(n)))
        k = min(self.recipe.readout.permutation_calls, n)
        return [tuple((i + shift) % n for i in range(n)) for shift in range(k)]

    async def _permuted(
        self,
        handle: StateHandle,
        question: Question,
        rung: Rung,
        slot: int | None,
        orders: list[tuple[int, ...]],
    ) -> RawAnswer:
        """Ask once per order and average the normalized distributions, so position bias
        cancels; the record says how many calls it took."""
        impl = RUNGS[rung]
        sums: dict[str, float] = dict.fromkeys(question.labels, 0.0)
        latency_ms = 0.0
        prompt_tokens = 0
        cached_tokens: int | None = None
        degraded = False
        off_menu: list[float] = []
        raw_parts = []
        for order in orders:
            part = render_question(self.recipe, question, order=order).parts[0]
            path, body = base_request(
                self.recipe.model.name, handle.prefix, part.text, self.dialect
            )
            impl.shape(body, part, self.recipe.readout.top_k, self.dialect)
            if slot is not None:
                body["id_slot"] = slot
            started = self.clock()
            post = self.client.post_root if path == "completion" else self.client.post_v1
            response = await post(path, body)
            latency_ms += (self.clock() - started) * 1000.0
            parsed = parse_response(path, response, self.dialect)
            readout = impl.read(parsed, part)
            distribution = Distribution.from_logprobs(readout.logprobs)
            for label, probability in distribution.as_mapping().items():
                sums[label] += probability
            degraded = degraded or readout.degraded
            if readout.off_menu_mass is not None:
                off_menu.append(readout.off_menu_mass)
            prompt_tokens += parsed.prompt_tokens or 0
            cached_tokens = parsed.cached_tokens
            raw_parts.append(
                {
                    "path": path,
                    "order": list(order),
                    "entries": [e.__dict__ for e in parsed.entries],
                }
            )
        k = len(orders)
        logprobs = {label: math.log(max(total / k, 1e-300)) for label, total in sums.items()}
        return RawAnswer(
            question_id=question.id,
            logprobs=logprobs,
            semantics="readout",
            rung=rung,
            degraded=degraded,
            off_menu_mass=(sum(off_menu) / len(off_menu)) if off_menu else None,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            calls=k,
            composition="permuted",
            raw={"parts": raw_parts},
        )

    async def _answer(
        self, handle: StateHandle, question: Question, rung: Rung, slot: int | None = None
    ) -> RawAnswer:
        """One request per part; a single-part question is the readout itself, a per-option
        question composes each option's yes-minus-no logit into one distribution."""
        rendered = render_question(self.recipe, question)
        orders = self._orders(question, rendered)
        if orders is not None:
            return await self._permuted(handle, question, rung, slot, orders)
        impl = RUNGS[rung]
        readouts = []
        latency_ms = 0.0
        prompt_tokens = 0
        cached_tokens: int | None = None
        raw_parts = []
        for part in rendered.parts:
            path, body = base_request(
                self.recipe.model.name, handle.prefix, part.text, self.dialect
            )
            impl.shape(body, part, self.recipe.readout.top_k, self.dialect)
            if slot is not None:
                body["id_slot"] = slot
            started = self.clock()
            post = self.client.post_root if path == "completion" else self.client.post_v1
            response = await post(path, body)
            latency_ms += (self.clock() - started) * 1000.0
            parsed = parse_response(path, response, self.dialect)
            readouts.append(impl.read(parsed, part))
            prompt_tokens += parsed.prompt_tokens or 0
            cached_tokens = parsed.cached_tokens
            raw_parts.append({"path": path, "entries": [e.__dict__ for e in parsed.entries]})
        if rendered.composition == "single":
            readout = readouts[0]
            logprobs, off_menu, missing = readout.logprobs, readout.off_menu_mass, readout.missing
        else:
            logprobs = {
                part.label: r.logprobs["true"] - r.logprobs["false"]
                for part, r in zip(rendered.parts, readouts, strict=True)
                if part.label is not None
            }
            off_menu, missing = None, ()
        return RawAnswer(
            question_id=question.id,
            logprobs=logprobs,
            semantics="readout",
            rung=rung,
            degraded=any(r.degraded for r in readouts),
            off_menu_mass=off_menu,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            missing=missing,
            calls=len(rendered.parts),
            composition=rendered.composition,
            raw={"parts": raw_parts},
        )
