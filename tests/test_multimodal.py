"""Multimodal state (ADR-0003 D5, design criterion 7): images as content parts, never URLs."""

from __future__ import annotations

import asyncio
import base64

import pytest
from starlette.testclient import TestClient

from jevify.api.app import create_app
from jevify.api.schema import SystemOneRequest
from jevify.api.translate import jev_request_to_domain
from jevify.compose import build_backend
from jevify.domain.engine import Engine
from jevify.domain.questions import ImagePart, NoulQuestion, State, TextPart
from jevify.ports.backend import BackendError, Capabilities, Dialect, Rung
from jevify.recipes import load_recipe
from tests.fakes.fake_backend import FakeBackend
from tests.fakes.fake_openai_server import Behaviour

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()
Q = NoulQuestion(id="red", instructions="The image is mostly red.")


def run(coro):
    return asyncio.run(coro)


def probed(recipe, modalities):
    caps = Capabilities(
        kind="endpoint",
        model=recipe.model.name,
        dialect=Dialect(recipe.endpoint.dialect),
        rungs=frozenset({Rung.TOP_K}),
        modalities=frozenset(modalities),
    )
    return recipe.model_copy(
        update={
            "probe": caps.to_dict(),
            "readout": recipe.readout.model_copy(update={"rung": "top_k"}),
        }
    )


def test_messages_mode_sends_image_parts_in_order(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(multimodal=True, scores={" yes": 0.0, " no": -1.0})
    )
    recipe = probed(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), {"text", "image"})
    backend = build_backend(recipe, http=http)
    state = State((TextPart("Screenshot of the checkout page."), ImagePart(PNG)))
    [answer] = run(backend.evaluate(run(backend.warm(state)), [Q]))
    content = server.requests[-1]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"] == PNG
    assert content[0]["text"].startswith("State:\nScreenshot")
    assert "mostly red" in content[2]["text"]
    assert set(answer.logprobs) == {"false", "true"}


def test_adapter_refuses_an_image_when_the_probe_found_no_image_support(
    fixtures, fake_server_factory
):
    server, http = fake_server_factory(Behaviour(scores={" yes": 0.0, " no": -1.0}))
    recipe = probed(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), {"text"})
    backend = build_backend(recipe, http=http)
    state = State((TextPart("x"), ImagePart(PNG)))
    with pytest.raises(BackendError, match="image"):
        run(backend.warm(state))
    assert server.requests == []  # refused before any request, never silently dropped


def test_jev_state_as_content_parts_and_urls_are_never_fetched():
    wire = SystemOneRequest.model_validate(
        {
            "state": [
                {"type": "text", "text": "Screenshot attached."},
                {"type": "image_url", "image_url": {"url": PNG}},
            ],
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }
    )
    state, _ = jev_request_to_domain(wire)
    assert state.parts == (TextPart("Screenshot attached."), ImagePart(PNG))
    assert state.modalities == {"text", "image"}
    remote = SystemOneRequest.model_validate(
        {
            "state": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }
    )
    with pytest.raises(ValueError, match="never fetches"):
        jev_request_to_domain(remote)


def test_api_rejects_remote_image_urls_with_422(fixtures):
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    app = create_app(Engine(FakeBackend({"q": {"false": 0.0, "true": -1.0}}), recipe), recipe)
    response = TestClient(app).post(
        "/v1/systemone",
        json={
            "state": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}],
            "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
    )
    assert response.status_code == 422
    assert "never fetches" in response.json()["error"]["message"]


def test_probe_detects_image_parts_accepted(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(multimodal=True, scores={" yes": 0.0, " no": -1.0}, max_context=100_000)
    )
    caps = run(
        build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http).probe()
    )
    assert "image" in caps.modalities
    server, http = fake_server_factory(
        Behaviour(multimodal=False, scores={" yes": 0.0, " no": -1.0}, max_context=100_000)
    )
    caps = run(
        build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http).probe()
    )
    assert "image" not in caps.modalities


def test_probe_reads_the_server_media_marker_and_writes_it_into_the_recipe(
    fixtures, fake_server_factory
):
    """llama.cpp draws a media marker per process and exposes it in /props; a prompt with
    any other marker is accepted with the image silently dropped."""
    from jevify.recipes.store import apply_probe

    server, http = fake_server_factory(
        Behaviour(
            dialect="llamacpp",
            multimodal=True,
            media_marker="<__media_k3j9__>",
            scores={"yes": 0.0, "no": -1.0},
            max_context=100_000,
        )
    )
    recipe = load_recipe(fixtures / "recipes" / "fake-raw.yaml")
    caps = run(build_backend(recipe, http=http).probe())
    assert "image" in caps.modalities
    assert caps.image_marker == "<__media_k3j9__>"
    assert any("LLAMA_MEDIA_MARKER" in note for note in caps.notes)
    assert apply_probe(recipe, caps).template.image_marker == "<__media_k3j9__>"


def test_probe_refuses_an_image_that_is_accepted_but_not_seen(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(
            multimodal=True, image_tokens=0, scores={" yes": 0.0, " no": -1.0}, max_context=100_000
        )
    )
    caps = run(
        build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http).probe()
    )
    assert "image" not in caps.modalities
    assert any("accepted but not seen" in note for note in caps.notes)
