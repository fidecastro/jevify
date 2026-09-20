"""The composition root (AGENTS.md §C "D"): the one place adapters are built and wired.

Tests inject an `httpx.AsyncClient` bound to a fake server; everything else
is the same code path as production.
"""

from __future__ import annotations

import os
from typing import Any

from jevify.adapters.endpoint.adapter import EndpointBackend
from jevify.adapters.endpoint.client import OpenAICompatibleClient
from jevify.domain.engine import Engine
from jevify.ports.backend import Backend, BackendError
from jevify.recipes.schema import Recipe


def build_engine(recipe: Recipe, *, http: Any | None = None) -> Engine:
    return Engine(build_backend(recipe, http=http), recipe)


def build_backend(recipe: Recipe, *, http: Any | None = None) -> Backend:
    if recipe.model.kind == "endpoint":
        assert recipe.endpoint is not None  # validated by the schema
        api_key = (
            os.environ.get(recipe.endpoint.api_key_env) if recipe.endpoint.api_key_env else None
        )
        client = OpenAICompatibleClient(
            recipe.endpoint.base_url,
            api_key=api_key,
            timeout_s=recipe.endpoint.timeout_s,
            http=http,
        )
        return EndpointBackend(client, recipe)
    raise BackendError(f"model kind {recipe.model.kind!r} has no adapter yet")
