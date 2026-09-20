"""The capability probe (ADR-0002 D3): a measurement, not a lookup table.

Every check sends real requests to the backend and records what came back.
The result is written into the recipe by `jevify probe`, and the adapter
picks its readout rung from it.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from jevify.adapters.endpoint.client import OpenAICompatibleClient
from jevify.adapters.endpoint.dialects import base_request, parse_response
from jevify.adapters.endpoint.readout import matches, read_answer_set
from jevify.domain.questions import NoulQuestion, State
from jevify.ports.backend import (
    BackendError,
    CacheEvidence,
    Capabilities,
    Dialect,
    Rung,
)
from jevify.recipes.render import RenderedPart, render_prefix, render_question
from jevify.recipes.schema import Recipe, Token

Clock = Callable[[], float]

_PROBE_CASE = Path(__file__).resolve().parents[2] / "recipes" / "probe_case.yaml"
_BLOCK_CANDIDATES = (1, 16, 32, 64, 128, 256, 512, 1024, 2048)


def load_probe_case() -> dict[str, Any]:
    return yaml.safe_load(_PROBE_CASE.read_text(encoding="utf-8"))


_CONTEXT_MARGIN = 160  # room for the template, the question and the answer slot
_CHARS_PER_TOKEN = 3.0  # conservative estimate when the backend cannot tokenize


class Prober:
    def __init__(
        self, client: OpenAICompatibleClient, recipe: Recipe, clock: Clock = time.perf_counter
    ):
        self.client = client
        self.recipe = recipe
        self.clock = clock
        self.notes: list[str] = []

    async def run(self) -> Capabilities:
        assert self.recipe.endpoint is not None
        case = load_probe_case()
        dialect, version = await self._dialect()
        max_context = await self._max_context(dialect)
        answer_tokens, multi, verified = await self._verify_tokens(dialect)
        kwargs_honored = await self._thinking_off(dialect)

        state = State.from_jev(await self._fit_state(case["state"], max_context, dialect))
        prefix = render_prefix(self.recipe, state)
        question = render_question(
            self.recipe, NoulQuestion(id="probe", instructions=case["statement"])
        )
        part = question.parts[0]

        parsed, entries_requested = await self._readout(
            prefix, part, dialect, self.recipe.readout.top_k
        )
        top_k_cap = len(parsed.entries) if parsed.entries else 0
        if top_k_cap == 0:
            raise BackendError("no readout is possible on this backend: it returned no logprobs")
        readout = read_answer_set(parsed, part, full_softmax=True)
        if readout.missing:
            self.notes.append(
                f"probe case: labels {list(readout.missing)} outside the top-{top_k_cap}"
            )

        rungs = {Rung.TOP_K, Rung.TOP_K_FLOOR}
        if await self._named_token_logprobs(prefix, part, dialect, answer_tokens):
            rungs.add(Rung.NAMED_TOKEN_LOGPROBS)
        post_bias = await self._bias_visible(prefix, part, dialect, parsed.entries, answer_tokens)
        if post_bias:
            rungs.add(Rung.EQUAL_BIAS)
        prefill = await self._prefill_honored(dialect) if prefix.mode == "messages" else None
        cache = await self._cache(state, dialect)

        return Capabilities(
            kind="endpoint",
            model=self.recipe.model.name,
            dialect=dialect,
            backend_version=version,
            max_context=max_context,
            modalities=frozenset({"text"}),
            rungs=frozenset(rungs),
            top_k_cap=top_k_cap if top_k_cap < entries_requested else None,
            logprobs_post_bias=post_bias,
            prefill_honored=prefill,
            template_kwargs_honored=kwargs_honored,
            token_ids_verified=verified,
            answer_tokens=answer_tokens,
            multi_token_answers=tuple(multi),
            cache=cache,
            probed_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            notes=tuple(self.notes),
        )

    # ------------------------------------------------------------------ identity
    async def _dialect(self) -> tuple[Dialect, str | None]:
        assert self.recipe.endpoint is not None
        pinned = self.recipe.endpoint.dialect
        for candidate, path, key in (
            (Dialect.VLLM, "version", "version"),
            (Dialect.LLAMACPP, "props", "build_info"),
        ):
            if pinned not in ("auto", str(candidate)):
                continue
            try:
                payload = await self.client.get_root(path)
            except BackendError:
                continue
            if isinstance(payload, dict) and key in payload:
                return candidate, str(payload[key])
        if pinned not in ("auto", "generic"):
            self.notes.append(f"dialect {pinned} pinned but its identity endpoint did not answer")
            return Dialect(pinned), None
        return Dialect.GENERIC, None

    async def _max_context(self, dialect: Dialect) -> int | None:
        """The context the server actually serves: llama.cpp's per-slot n_ctx outranks the
        model's trained context, which /v1/models reports as meta.n_ctx_train."""
        if dialect is Dialect.LLAMACPP:
            try:
                props = await self.client.get_root("props")
                n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
                if n_ctx:
                    return int(n_ctx)
            except BackendError:
                pass
        try:
            models = await self.client.get_v1("models")
        except BackendError:
            models = {}
        for entry in models.get("data") or []:
            if entry.get("max_model_len"):
                return int(entry["max_model_len"])
            meta = entry.get("meta") or {}
            if meta.get("n_ctx_train"):
                self.notes.append(
                    "max_context is the trained context; the served context is unknown"
                )
                return int(meta["n_ctx_train"])
        return None

    # ------------------------------------------------------------------ tokens
    async def _tokenize(self, text: str, dialect: Dialect) -> list[int] | None:
        try:
            if dialect is Dialect.LLAMACPP:
                payload = await self.client.post_root(
                    "tokenize", {"content": text, "add_special": False, "parse_special": False}
                )
                tokens = payload.get("tokens") or []
                return [t["id"] if isinstance(t, dict) else int(t) for t in tokens]
            payload = await self.client.post_root(
                "tokenize",
                {"model": self.recipe.model.name, "prompt": text, "add_special_tokens": False},
            )
            return [int(t) for t in payload.get("tokens") or []]
        except BackendError:
            return None

    async def _verify_tokens(self, dialect: Dialect) -> tuple[dict[str, int], list[str], bool]:
        groups: list[list[Token]] = [
            *self.recipe.answers.noul.values(),
            *self.recipe.answers.identifiers,
        ]
        texts = list(dict.fromkeys(token.text for group in groups for token in group))
        ids: dict[str, int] = {}
        multi: list[str] = []
        for text in texts:
            tokens = await self._tokenize(text, dialect)
            if tokens is None:
                self.notes.append("no tokenize endpoint: answer token ids are unverified")
                return {}, [], False
            if len(tokens) == 1:
                ids[text] = tokens[0]
            else:
                multi.append(text)
        return ids, multi, True

    async def _thinking_off(self, dialect: Dialect) -> bool | None:
        """Render the probe prompt through the server's own template and look for a think marker."""
        if self.recipe.template.mode != "messages":
            return None
        messages: list[dict[str, Any]] = []
        if self.recipe.template.system:
            messages.append({"role": "system", "content": self.recipe.template.system})
        messages.append({"role": "user", "content": "probe"})
        rendered: str | None = None
        try:
            if dialect is Dialect.LLAMACPP:
                payload = await self.client.post_root(
                    "apply-template",
                    {
                        "messages": messages,
                        "chat_template_kwargs": dict(self.recipe.template.kwargs),
                    },
                )
                rendered = str(payload.get("prompt", ""))
            else:
                payload = await self.client.post_root(
                    "tokenize",
                    {
                        "model": self.recipe.model.name,
                        "messages": messages,
                        "add_generation_prompt": True,
                        "return_token_strs": True,
                        "chat_template_kwargs": dict(self.recipe.template.kwargs),
                    },
                )
                rendered = "".join(payload.get("token_strs") or [])
        except BackendError:
            self.notes.append("could not render the template server-side; thinking state unknown")
            return None
        tail = rendered.rstrip()[-40:]
        markers = load_probe_case().get("think_markers") or []
        return not any(marker in tail for marker in markers)

    # ------------------------------------------------------------------ readout checks
    async def _readout(self, prefix, part: RenderedPart, dialect: Dialect, top_k: int):
        path, body = base_request(self.recipe.model.name, prefix, part.text, dialect)
        if "messages" in body:
            body["top_logprobs"] = top_k
        else:
            body["logprobs"] = top_k
        response = await self.client.post_v1(path, body)
        return parse_response(path, response, dialect), top_k

    async def _named_token_logprobs(self, prefix, part, dialect, answer_tokens) -> bool:
        if dialect is not Dialect.VLLM or not answer_tokens:
            return False
        ids = sorted(
            {
                answer_tokens[t.text]
                for lbl in part.labels
                for t in part.tokens[lbl]
                if t.text in answer_tokens
            }
        )
        path, body = base_request(self.recipe.model.name, prefix, part.text, dialect)
        body.update({"top_logprobs": 1, "logprob_token_ids": ids, "return_as_token_id": True})
        try:
            response = await self.client.post_v1(path, body)
        except BackendError as exc:
            self.notes.append(f"logprob_token_ids rejected: {exc}")
            return False
        parsed = parse_response(path, response, dialect)
        returned = {e.id for e in parsed.entries if e.id is not None}
        return set(ids) <= returned

    async def _bias_visible(self, prefix, part, dialect, entries, answer_tokens) -> bool | None:
        """Bias the weakest answer token by +50 and see whether its reported logprob moves."""
        candidates = [
            e
            for e in entries
            if any(matches(e, t) for lbl in part.labels for t in part.tokens[lbl])
        ]
        if not candidates:
            return None
        weakest = min(candidates, key=lambda e: e.logprob)
        ident = weakest.id if weakest.id is not None else answer_tokens.get(weakest.text)
        if ident is None:
            return None
        path, body = base_request(self.recipe.model.name, prefix, part.text, dialect)
        if "messages" in body:
            body["top_logprobs"] = self.recipe.readout.top_k
        else:
            body["logprobs"] = self.recipe.readout.top_k
        body["logit_bias"] = {str(ident): 50}
        try:
            response = await self.client.post_v1(path, body)
        except BackendError as exc:
            self.notes.append(f"logit_bias rejected: {exc}")
            return None
        parsed = parse_response(path, response, dialect)
        for entry in parsed.entries:
            if (entry.id == ident) or (entry.id is None and entry.text == weakest.text):
                return entry.logprob > weakest.logprob + 1.0
        return False

    async def _prefill_honored(self, dialect: Dialect) -> bool | None:
        if not self.recipe.template.prefill:
            return None
        messages: list[dict[str, Any]] = [{"role": "user", "content": "probe"}]
        messages.append({"role": "assistant", "content": self.recipe.template.prefill})
        try:
            payload = await self.client.post_root(
                "tokenize",
                {
                    "model": self.recipe.model.name,
                    "messages": messages,
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                    "return_token_strs": True,
                },
            )
        except BackendError:
            return None
        rendered = "".join(payload.get("token_strs") or [])
        return rendered.rstrip().endswith(self.recipe.template.prefill)

    async def _fit_state(self, text: str, max_context: int | None, dialect: Dialect) -> str:
        """Cut the probe state so the full prompt fits the backend's context (INV-9 says the
        adapter must never truncate a user's state; the probe's own case is ours to size)."""
        if max_context is None:
            return text
        budget = max_context - _CONTEXT_MARGIN
        if budget <= 0:
            raise BackendError(f"context of {max_context} tokens is too small to probe")
        tokens = await self._tokenize(text, dialect)
        count = len(tokens) if tokens is not None else int(len(text) / _CHARS_PER_TOKEN)
        if count <= budget:
            return text
        keep = int(len(text) * budget / count)
        self.notes.append(f"probe state cut to about {budget} tokens to fit the context")
        return text[:keep]

    # ------------------------------------------------------------------ cache
    async def _cache(self, state: State, dialect: Dialect) -> CacheEvidence:
        """Send identical readouts over three prefix lengths; read the cached-token count."""
        text = state.text
        observations: list[tuple[int, int | None]] = []
        first_ms = second_ms = None
        for fraction in (0.35, 0.7, 1.0):
            sub = State.from_jev(text[: int(len(text) * fraction)])
            prefix = render_prefix(self.recipe, sub)
            part = render_question(
                self.recipe, NoulQuestion(id="cache", instructions="cache probe")
            ).parts[0]
            path, body = base_request(self.recipe.model.name, prefix, part.text, dialect)
            body["logprobs"] = True if "messages" in body else 1
            if "messages" in body:
                body["top_logprobs"] = 1
            t0 = self.clock()
            first = parse_response(path, await self.client.post_v1(path, body), dialect)
            t1 = self.clock()
            second = parse_response(path, await self.client.post_v1(path, body), dialect)
            t2 = self.clock()
            if fraction == 1.0:
                first_ms, second_ms = (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
            if first.prompt_tokens is not None:
                observations.append((first.prompt_tokens, second.cached_tokens))
        last_cached = observations[-1][1] if observations else None
        if last_cached is None:
            self.notes.append(
                "backend reports no cached-token count; cache evidence is timing only"
            )
            return CacheEvidence(None, first_ms, second_ms, None)
        consistent = [
            b
            for b in _BLOCK_CANDIDATES
            if all(
                (c is not None) and ((c >= n - 1) if b == 1 else (c == b * (n // b)))
                for n, c in observations
            )
        ]
        block = consistent[0] if consistent else None
        if block is None:
            self.notes.append(f"cached-token counts {observations} fit no block size candidate")
        return CacheEvidence(block, first_ms, second_ms, last_cached)
