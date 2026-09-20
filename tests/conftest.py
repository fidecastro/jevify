"""Shared fixtures: fake servers served in-process through httpx's ASGI transport."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests.fakes.fake_openai_server import Behaviour, FakeOpenAIServer

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def fake_server_factory() -> Callable[
    [Behaviour | None], tuple[FakeOpenAIServer, httpx.AsyncClient]
]:
    """Build a fake server and an httpx client that talks to it without sockets."""

    def make(behaviour: Behaviour | None = None) -> tuple[FakeOpenAIServer, httpx.AsyncClient]:
        server = FakeOpenAIServer(behaviour)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://fake"
        )
        return server, client

    return make
