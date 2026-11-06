"""The prompt_logprobs rung in messages mode (ADR-0002 D4, amended 2026-11-06).

TensorFold's native server takes a chat `messages` list on /v1/completions too, rendered
through the model's chat template with its generation prompt, and serves `prompt_logprobs`
there. That is the only route on which such a server carries images to a readout. The rung
appends the answer as a final assistant message and reads that token's row. The template
then opens a new assistant turn after it, so the answer's row is not the last one: the probe
measures how many rows follow it, and every readout checks the row it reads holds the answer.

The fake renders `<|role|>` markers and content as pieces; with thinking on (the fake's
default for a dialect that sends no template arguments) the generation prompt is
`<|assistant|><think>`, two rows after the answer. Expected values are literals derived from
the fake's logit table.
"""

from __future__ import annotations

import asyncio
import base64
import math

import pytest

from jevify.compose import build_backend
from jevify.domain.distribution import Distribution
from jevify.domain.questions import ImagePart, NoulQuestion, State, TextPart
from jevify.ports.backend import BackendError, Rung
from jevify.recipes import apply_probe, load_recipe
from tests.fakes.fake_openai_server import Behaviour

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()
STATE = State.from_jev("The customer wrote: keep my subscription, stop the newsletters.")
Q = NoulQuestion(id="q", instructions="The customer wants to cancel.")
# TensorFold with GLM-5.3-Flash, vision on: no reply-token logprobs, prompt_logprobs on
# /v1/completions without prefix reuse, a chat `messages` list accepted there too
TENSORFOLD = {
    "dialect": "generic",
    "reply_logprobs": False,
    "prompt_logprobs": True,
    "prompt_logprobs_reuse": False,
    "completions_messages": True,
    "multimodal": True,
    "max_context": 100_000,
}


def run(coro):
    return asyncio.run(coro)


def probed(fixtures, http):
    recipe = load_recipe(fixtures / "recipes" / "fake-tensorfold-messages.yaml")
    return apply_probe(recipe, run(build_backend(recipe, http=http).probe()))


def readouts(server):
    return [r for r in server.requests if "prompt_logprobs" in r]


def test_probe_proves_the_rung_on_completions_messages_and_measures_the_rows_after(
    fixtures, fake_server_factory
):
    server, http = fake_server_factory(Behaviour(scores={"yes": 0.0, "no": -1.0}, **TENSORFOLD))
    recipe = load_recipe(fixtures / "recipes" / "fake-tensorfold-messages.yaml")
    caps = run(build_backend(recipe, http=http).probe())
    assert caps.rungs == frozenset({Rung.PROMPT_LOGPROBS})
    # the fake's generation prompt after the appended answer: <|assistant|> and <think>
    assert caps.prompt_rows_after_answer == 2
    # proven on the route the readout uses: the image adds prompt tokens there
    assert "image" in caps.modalities
    images = [
        r
        for r in server.requests
        if any(isinstance(m["content"], list) for m in r.get("messages", []))
    ]
    assert images and all(r["path"] == "/v1/completions" for r in images)


def test_probe_leaves_the_rung_unproven_where_completions_takes_no_messages(
    fixtures, fake_server_factory
):
    server, http = fake_server_factory(
        Behaviour(scores={"yes": 0.0}, **{**TENSORFOLD, "completions_messages": False})
    )
    recipe = load_recipe(fixtures / "recipes" / "fake-tensorfold-messages.yaml")
    with pytest.raises(BackendError, match="no readout"):
        run(build_backend(recipe, http=http).probe())


def test_reads_the_answer_row_not_the_last_row(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={"yes": 0.0, "no": -1.0}, **TENSORFOLD))
    backend = build_backend(probed(fixtures, http), http=http)
    server.requests.clear()
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "prompt_logprobs"
    assert answer.degraded is False and answer.missing == ()
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] == pytest.approx(1 / (1 + math.exp(-1)))
    # yes at 0, no at -1, five fillers at -3: the fillers are the off-menu mass
    z = 1 + math.exp(-1) + 5 * math.exp(-3)
    assert answer.off_menu_mass == pytest.approx(5 * math.exp(-3) / z)
    [request] = readouts(server)
    assert request["path"] == "/v1/completions"
    assert request["messages"][-1] == {"role": "assistant", "content": "no"}
    # nothing the server refuses: no reply-token logprobs, no chat route, warm included
    assert all(r["path"] == "/v1/completions" for r in server.requests)
    assert all("logprobs" not in r and "top_logprobs" not in r for r in server.requests)
    assert answer.calls == 1


def test_an_image_in_the_state_reaches_the_readout(fixtures, fake_server_factory):
    # the "model" answers differently when an image is in its prompt
    behaviour = Behaviour(
        scores={"yes": 0.0, "no": -1.0},
        scores_when=[("<image>", {"yes": 0.0, "no": -4.0})],
        **TENSORFOLD,
    )
    server, http = fake_server_factory(behaviour)
    backend = build_backend(probed(fixtures, http), http=http)
    server.requests.clear()
    state = State((TextPart("Screenshot of the checkout page."), ImagePart(PNG)))
    [answer] = run(backend.evaluate(run(backend.warm(state)), [Q]))
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] == pytest.approx(1 / (1 + math.exp(-4)))
    [request] = readouts(server)
    content = request["messages"][-2]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"] == PNG


def test_a_row_count_the_server_no_longer_matches_fails_loudly(fixtures, fake_server_factory):
    # probed with thinking on (two rows after the answer); the server now renders one
    server, http = fake_server_factory(Behaviour(scores={"yes": 0.0, "no": -1.0}, **TENSORFOLD))
    recipe = probed(fixtures, http)
    server.behaviour.thinking_default_on = False
    backend = build_backend(recipe, http=http)
    with pytest.raises(BackendError, match="re-run the probe"):
        run(backend.evaluate(run(backend.warm(STATE)), [Q]))


def test_a_messages_recipe_without_the_measured_row_count_is_refused(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={"yes": 0.0, "no": -1.0}, **TENSORFOLD))
    recipe = probed(fixtures, http)
    probe = {**recipe.probe, "prompt_rows_after_answer": None}
    backend = build_backend(recipe.model_copy(update={"probe": probe}), http=http)
    server.requests.clear()
    with pytest.raises(BackendError, match="prompt_rows_after_answer"):
        run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert readouts(server) == []
