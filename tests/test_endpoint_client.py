"""The HTTP client (`client.py`, the only module that imports httpx): static headers
a recipe asks for on every request, and the guard against overriding what jevify sets."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from jevify.adapters.endpoint.client import OpenAICompatibleClient
from jevify.compose import build_backend
from jevify.ports.backend import BackendError
from jevify.recipes import load_recipe

RECIPES = Path(__file__).parent / "fixtures" / "recipes"


def run(coro):
    return asyncio.run(coro)


def _client(**kwargs) -> tuple[OpenAICompatibleClient, dict[str, httpx.Headers]]:
    captured: dict[str, httpx.Headers] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"object": "list", "data": []})

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://fake")
    client = OpenAICompatibleClient("http://fake/v1", http=http, **kwargs)
    return client, captured


def test_extra_headers_are_sent_on_every_request() -> None:
    client, captured = _client(headers={"x-client-id": "agenticApp"})
    run(client.get_v1("models"))
    assert captured["headers"]["x-client-id"] == "agenticApp"


def test_extra_headers_coexist_with_bearer_auth() -> None:
    client, captured = _client(headers={"x-client-id": "agenticApp"}, api_key="secret")
    run(client.get_v1("models"))
    assert captured["headers"]["x-client-id"] == "agenticApp"
    assert captured["headers"]["authorization"] == "Bearer secret"


@pytest.mark.parametrize("fixture", ["fake-vllm.yaml", "fake-rerank.yaml", "fake-embedding.yaml"])
def test_every_endpoint_backed_kind_sends_the_recipe_headers(fixture: str) -> None:
    seen: list[httpx.Headers] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(400, json={"error": {"message": "refused by the test"}})

    recipe = load_recipe(RECIPES / fixture)
    recipe.endpoint.headers = {"x-client-id": "agenticApp"}
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake")
    with pytest.raises(BackendError):
        run(build_backend(recipe, http=http).probe())

    assert seen
    assert all(h.get("x-client-id") == "agenticApp" for h in seen)


@pytest.mark.parametrize("name", ["Authorization", "Content-Type", "authorization"])
def test_endpoint_headers_may_not_override_what_jevify_sets(name: str) -> None:
    with pytest.raises(BackendError, match="headers"):
        OpenAICompatibleClient("http://fake/v1", headers={name: "x"})
