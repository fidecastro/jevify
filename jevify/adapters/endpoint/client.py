"""The only module that imports httpx. Timeouts, bounded retries, error wrapping."""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Mapping
from typing import Any

import httpx

from jevify.ports.backend import BackendError, ContextLimitError

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_CONTEXT_LIMIT = re.compile(
    r"(maximum context length|context length|too many tokens|exceeds)", re.I
)


# Every top-level request field jevify itself sets: the readout requests of every dialect
# and rung, the probe's measurements, the tokenize and template routes, and the rerank and
# embeddings bodies. `endpoint.extra_body` may not name one of these; the client refuses
# when it is built, before any request, so probe, ask, eval and serve all fail the same way.
# Guarded by tests/guards/test_request_fields.py, which records every field the adapters
# send to the fakes and fails naming any that is missing here.
RESERVED_REQUEST_FIELDS: frozenset[str] = frozenset(
    {
        # the readout request (dialects.base_request) and prefill
        "model",
        "messages",
        "prompt",
        "max_tokens",
        "n_predict",
        "temperature",
        "stream",
        "logprobs",
        "continue_final_message",
        "add_generation_prompt",
        "chat_template_kwargs",
        # the rungs and the probe
        "top_logprobs",
        "logprob_token_ids",
        "return_as_token_id",
        "logit_bias",
        "grammar",
        "n_probs",
        # llama.cpp neutral samplers and slot pinning
        "top_k",
        "top_p",
        "min_p",
        "post_sampling_probs",
        "id_slot",
        # tokenize and apply-template
        "content",
        "add_special",
        "parse_special",
        "add_special_tokens",
        "return_token_strs",
        # rerank and embeddings
        "query",
        "documents",
        "top_n",
        "instruction",
        "input",
    }
)


class OpenAICompatibleClient:
    """HTTP access to one server. `v1/*` for the OpenAI surface, root for server extras."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        http: httpx.AsyncClient | None = None,
        max_retries: int = 3,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        stripped = base_url.rstrip("/")
        self.root = stripped[: -len("/v1")] if stripped.endswith("/v1") else stripped
        self.v1 = self.root + "/v1"
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        _refuse_reserved(set(self.extra_body) & RESERVED_REQUEST_FIELDS)
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(self, method: str, url: str, body: dict[str, Any] | None = None) -> Any:
        if body is not None and self.extra_body:
            body = _merge_extra(body, self.extra_body)
        attempt = 0
        while True:
            try:
                response = await self._http.request(
                    method, url, json=body, headers=self._headers(), timeout=self.timeout_s
                )
            except httpx.HTTPError as exc:
                raise BackendError(f"{method} {url} failed: {exc}") from exc
            if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                attempt += 1
                await asyncio.sleep(_retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise _http_error(method, url, response)
            return response.json()

    async def post_v1(self, path: str, body: dict[str, Any]) -> Any:
        return await self._request("POST", f"{self.v1}/{path.lstrip('/')}", body)

    async def get_v1(self, path: str) -> Any:
        return await self._request("GET", f"{self.v1}/{path.lstrip('/')}")

    async def post_root(self, path: str, body: dict[str, Any]) -> Any:
        return await self._request("POST", f"{self.root}/{path.lstrip('/')}", body)

    async def get_root(self, path: str) -> Any:
        return await self._request("GET", f"{self.root}/{path.lstrip('/')}")

    async def aclose(self) -> None:
        await self._http.aclose()


def _refuse_reserved(clash: set[str]) -> None:
    if clash:
        raise BackendError(
            f"endpoint.extra_body may not set {', '.join(sorted(clash))}: jevify sets it"
        )


def _merge_extra(body: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Add the recipe's extra fields; never let one replace a field jevify set. The registry
    check at construction is the rule; this per-request check is the backstop for a field
    the registry does not know yet."""
    _refuse_reserved(set(body) & set(extra))
    return {**body, **extra}


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after-ms") or response.headers.get("retry-after")
    if header:
        try:
            value = float(header)
            return (
                value / 1000.0 if "ms" in (response.headers.get("retry-after-ms") or "") else value
            )
        except ValueError:
            pass
    return min(2.0, 0.05 * (2**attempt)) + random.uniform(0.0, 0.05)  # noqa: S311 - jitter only


def _http_error(method: str, url: str, response: httpx.Response) -> BackendError:
    message = _error_message(response)
    text = f"{method} {url} returned {response.status_code}: {message}"
    if response.status_code in (400, 413, 422) and _CONTEXT_LIMIT.search(message):
        return ContextLimitError(text, status=response.status_code)
    return BackendError(text, status=response.status_code)


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
        if isinstance(error, str):
            return error
        for key in ("message", "detail"):
            if key in body:
                return str(body[key])
    return str(body)[:500]
