"""The HTTP server: Jev's evaluate call, unchanged, over the library (ADR-0003 D1, D7).

Thin by design. Validation is explicit pydantic; errors use the FastAPI-style
`{"detail": [...]}` body the SDK understands; every response carries an
`x-typesafe-request-id` header because the SDK expects one.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from jevify import __version__
from jevify.api.schema import HTTPValidationError, SystemOneRequest, SystemOneResponse
from jevify.api.translate import (
    classify_results,
    classify_to_systemone,
    evaluation_to_jev,
    jev_extension,
    jev_request_to_domain,
)
from jevify.domain.engine import Engine
from jevify.ports.backend import BackendError, ContextLimitError
from jevify.recipes.schema import Recipe
from jevify.recipes.store import RecipeError, recipe_hash

REQUEST_ID_HEADER = "x-typesafe-request-id"


def _validation_error(exc: ValidationError) -> JSONResponse:
    detail = [
        {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
        for item in exc.errors()
    ]
    body = HTTPValidationError(detail=detail).model_dump(mode="json", exclude_none=True)
    return JSONResponse(body, status_code=422)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "jevify_error"}}, status_code=status)


@dataclass
class _HeldState:
    handle: Any
    expires_at: float


class StateStore:
    """In-process handles for the optional ingest route (ADR-0003 D4): one process, a TTL,
    and a loud 404 for an expired or unknown handle."""

    def __init__(self, ttl_s: float, clock=time.monotonic) -> None:
        self.ttl_s = ttl_s
        self.clock = clock
        self._held: dict[str, _HeldState] = {}

    def put(self, handle: Any) -> str:
        self._sweep()
        self._held[handle.state_id] = _HeldState(handle, self.clock() + self.ttl_s)
        return handle.state_id

    def get(self, state_id: str) -> Any | None:
        self._sweep()
        held = self._held.get(state_id)
        return held.handle if held else None

    def _sweep(self) -> None:
        now = self.clock()
        for key in [k for k, v in self._held.items() if v.expires_at <= now]:
            del self._held[key]


def create_app(
    engine: Engine, recipe: Recipe, *, api_key: str | None = None, state_ttl_s: float = 600.0
) -> Starlette:
    digest = recipe_hash(recipe)
    model_name = recipe.model.name
    states = StateStore(state_ttl_s)

    def authorized(request: Request) -> bool:
        if api_key is None:
            return True
        header = request.headers.get("authorization", "")
        return header == f"Bearer {api_key}"

    async def systemone(request: Request) -> JSONResponse:
        if not authorized(request):
            return _error(401, "invalid api key")
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "body is not valid JSON")
        return await evaluate_payload(payload)

    async def evaluate_payload(payload: Any) -> JSONResponse:
        """The one evaluation path: systemone and classify both end here."""
        try:
            wire = SystemOneRequest.model_validate(payload)
        except ValidationError as exc:
            return _validation_error(exc)
        try:
            state, questions = jev_request_to_domain(wire)
            state_ref = (wire.x_jevify or {}).get("state_ref")
            if state_ref:
                handle = states.get(str(state_ref))
                if handle is None:
                    return _error(404, f"state handle {state_ref!r} is expired or unknown")
                evaluation = await engine.evaluate(handle, questions)
            else:
                evaluation = await engine.ask(state, questions)
        except ContextLimitError as exc:
            return _error(413, str(exc))
        except (BackendError, RecipeError) as exc:
            return _error(502, str(exc))
        body: dict[str, Any] = evaluation_to_jev(evaluation, model=model_name, recipe_hash=digest)
        body["x_jevify"] = jev_extension(
            evaluation, {"recipe_hash": digest, "requested_model": wire.model}
        )
        SystemOneResponse.model_validate(body)  # never emit a shape the SDK would reject
        return JSONResponse(body, headers={REQUEST_ID_HEADER: f"req_{uuid.uuid4().hex}"})

    async def create_state(request: Request) -> JSONResponse:
        if not authorized(request):
            return _error(401, "invalid api key")
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "body is not valid JSON")
        if not isinstance(payload, dict) or payload.get("state") is None:
            return _error(422, "a state is required")
        try:
            wire = SystemOneRequest.model_validate(
                {
                    "state": payload["state"],
                    "model": model_name,
                    "questions": {"_": {"type": "noul"}},
                }
            )
            state, _ = jev_request_to_domain(wire)
            handle = await engine.warm(state)
        except ValidationError as exc:
            return _validation_error(exc)
        except ContextLimitError as exc:
            return _error(413, str(exc))
        except (BackendError, RecipeError) as exc:
            return _error(502, str(exc))
        state_id = states.put(handle)
        return JSONResponse(
            {
                "state_id": state_id,
                "expires_in_s": state_ttl_s,
                "warm_ms": handle.warm_ms,
                "cached_tokens": handle.cached_tokens,
            },
            status_code=201,
            headers={REQUEST_ID_HEADER: f"req_{uuid.uuid4().hex}"},
        )

    async def classify(request: Request) -> JSONResponse:
        if not authorized(request):
            return _error(401, "invalid api key")
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "body is not valid JSON")
        try:
            jev_request = classify_to_systemone(payload)
        except ValueError as exc:
            return _error(422, str(exc))
        response = await evaluate_payload(jev_request)
        if response.status_code != 200:
            return response
        body = json.loads(bytes(response.body))
        mode = payload.get("mode", "boolean")
        categories = (
            list(jev_request["questions"])
            if mode != "choice"
            else list(jev_request["questions"]["category"]["criteria"])
        )
        body["results"] = classify_results(mode, categories, body["answers"])
        body["mode"] = mode
        return JSONResponse(
            body, headers={REQUEST_ID_HEADER: response.headers.get(REQUEST_ID_HEADER, "")}
        )

    async def models(request: Request) -> JSONResponse:
        if not authorized(request):
            return _error(401, "invalid api key")
        return JSONResponse(
            {
                "models": [
                    {
                        "name": model_name,
                        "description": f"jevify {__version__} recipe {digest[:12]}",
                        "release_date": recipe.provenance.created,
                    }
                ]
            },
            headers={REQUEST_ID_HEADER: f"req_{uuid.uuid4().hex}"},
        )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "model": model_name, "recipe_hash": digest})

    return Starlette(
        routes=[
            Route("/v1/systemone", systemone, methods=["POST"]),
            Route("/v1/states", create_state, methods=["POST"]),
            Route("/v1/classify", classify, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ]
    )
