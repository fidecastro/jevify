"""The HTTP server: Jev's evaluate call, unchanged, over the library (ADR-0003 D1, D7).

Thin by design. Validation is explicit pydantic; errors use the FastAPI-style
`{"detail": [...]}` body the SDK understands; every response carries an
`x-typesafe-request-id` header because the SDK expects one.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from jevify import __version__
from jevify.api.schema import HTTPValidationError, SystemOneRequest, SystemOneResponse
from jevify.api.translate import evaluation_to_jev, jev_extension, jev_request_to_domain
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


def create_app(engine: Engine, recipe: Recipe, *, api_key: str | None = None) -> Starlette:
    digest = recipe_hash(recipe)
    model_name = recipe.model.name

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
        try:
            wire = SystemOneRequest.model_validate(payload)
        except ValidationError as exc:
            return _validation_error(exc)
        try:
            state, questions = jev_request_to_domain(wire)
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
            Route("/v1/models", models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ]
    )
