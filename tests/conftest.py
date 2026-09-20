"""Shared fixtures: fake servers served in-process through httpx's ASGI transport."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import uvicorn

from tests.fakes.fake_openai_server import Behaviour, FakeOpenAIServer

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@dataclass
class LiveFake:
    """A fake server bound to a real localhost port, for subprocess- and SDK-level tests."""

    server: FakeOpenAIServer
    base_url: str


@pytest.fixture
def fake_http_server_factory() -> Iterator[Callable[[Behaviour | None], LiveFake]]:
    """Start a fake server on a real port in a thread; stop it at teardown."""
    running: list[tuple[uvicorn.Server, threading.Thread]] = []

    def make(behaviour: Behaviour | None = None) -> LiveFake:
        fake = FakeOpenAIServer(behaviour)
        config = uvicorn.Config(fake.app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("fake server did not start")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        running.append((server, thread))
        return LiveFake(fake, f"http://127.0.0.1:{port}/v1")

    yield make
    for server, thread in running:
        server.should_exit = True
        thread.join(timeout=5)


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
