"""The prompt_logprobs rung (ADR-0002 D4, amended 2026-11-04).

A server that refuses reply-token logprobs (TensorFold's native server with GLM-5.3-Flash)
can still serve vLLM's ``prompt_logprobs`` on /v1/completions. The rung appends an answer
token to a raw prompt and reads that position's row back: the appended token's exact logprob
and the top-k beside it. Expected values are literals derived from the fake's logit table.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from jevify.compose import build_backend
from jevify.domain.distribution import Distribution
from jevify.domain.questions import NoulQuestion, State
from jevify.ports.backend import BackendError, Capabilities, Dialect, Rung
from jevify.recipes import load_recipe
from tests.fakes.fake_openai_server import Behaviour

STATE = State.from_jev("The customer wrote: keep my subscription, stop the newsletters.")
Q = NoulQuestion(id="q", instructions="The customer wants to cancel.")
# TensorFold with GLM-5.3-Flash: no reply-token logprobs, prompt_logprobs without prefix reuse
TENSORFOLD = {
    "dialect": "generic",
    "reply_logprobs": False,
    "prompt_logprobs": True,
    "prompt_logprobs_reuse": False,
    "max_context": 100_000,
}


def run(coro):
    return asyncio.run(coro)


def recipe_with(fixtures, rungs, rung="auto", name="fake-tensorfold.yaml"):
    recipe = load_recipe(fixtures / "recipes" / name)
    caps = Capabilities(
        kind="endpoint",
        model=recipe.model.name,
        dialect=Dialect(recipe.endpoint.dialect),
        rungs=frozenset(Rung(r) for r in rungs),
        token_ids_verified=True,
    )
    return recipe.model_copy(
        update={
            "probe": caps.to_dict(),
            "readout": recipe.readout.model_copy(update={"rung": rung}),
        }
    )


def test_probe_proves_prompt_logprobs_where_reply_logprobs_are_refused(
    fixtures, fake_server_factory
):
    server, http = fake_server_factory(Behaviour(scores={"yes": 0.0, "no": -1.0}, **TENSORFOLD))
    recipe = load_recipe(fixtures / "recipes" / "fake-tensorfold.yaml")
    caps = run(build_backend(recipe, http=http).probe())
    assert caps.rungs == frozenset({Rung.PROMPT_LOGPROBS})
    assert caps.top_k_cap is None
    assert any("reply-token logprobs refused" in note for note in caps.notes)
    # measured, not assumed: a repeated prompt_logprobs request was served without the cache
    assert any("reuse no cached prefix" in note for note in caps.notes)
    # no probe request after the refusal asked for reply-token logprobs again
    refused_at = next(
        i for i, r in enumerate(server.requests) if r.get("logprobs") not in (None, False)
    )
    assert all(r.get("logprobs") in (None, False) for r in server.requests[refused_at + 1 :])


def test_probe_refuses_when_neither_reply_nor_prompt_logprobs_come_back(
    fixtures, fake_server_factory
):
    server, http = fake_server_factory(
        Behaviour(scores={"yes": 0.0}, **{**TENSORFOLD, "prompt_logprobs": False})
    )
    recipe = load_recipe(fixtures / "recipes" / "fake-tensorfold.yaml")
    with pytest.raises(BackendError, match="no readout"):
        run(build_backend(recipe, http=http).probe())


def test_reads_the_exact_distribution_with_off_menu_mass(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={"yes": 0.0, "no": -1.0}, **TENSORFOLD))
    backend = build_backend(recipe_with(fixtures, ["prompt_logprobs"]), http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "prompt_logprobs"
    assert answer.degraded is False and answer.missing == ()
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] == pytest.approx(1 / (1 + math.exp(-1)))
    # yes at 0, no at -1, five fillers at -3: the fillers are the off-menu mass
    z = 1 + math.exp(-1) + 5 * math.exp(-3)
    assert answer.off_menu_mass == pytest.approx(5 * math.exp(-3) / z)
    [request] = [r for r in server.requests if "prompt_logprobs" in r]
    assert request["path"] == "/v1/completions"
    assert request["prompt"].endswith("\n\nno")  # the first label's spelling, appended
    assert "logprobs" not in request
    assert answer.calls == 1


def test_a_spelling_outside_the_top_k_is_appended_in_a_request_of_its_own(
    fixtures, fake_server_factory
):
    # The server lists one alternative per row, and "yes" is far below "no": the row of the
    # request that appends "no" cannot show "yes", so "yes" is appended and read exactly.
    server, http = fake_server_factory(
        Behaviour(scores={"yes": -9.0, "no": 0.0}, top_k_cap=1, **TENSORFOLD)
    )
    backend = build_backend(recipe_with(fixtures, ["prompt_logprobs"]), http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.calls == 2
    assert [r["prompt"][-3:] for r in server.requests if "prompt_logprobs" in r] == ["\nno", "yes"]
    assert answer.degraded is False and answer.missing == ()
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] == pytest.approx(math.exp(-9) / (1 + math.exp(-9)))


def test_auto_takes_prompt_logprobs_before_the_floor(fixtures, fake_server_factory):
    """Placed after top_k: when top_k cannot see a label, prompt_logprobs reads it exactly
    instead of the floor flooring it."""
    server, http = fake_server_factory(
        Behaviour(
            scores={"yes": 0.0, "no": -9.0},
            top_k_cap=5,
            dialect="generic",
            prompt_logprobs=True,
            max_context=100_000,
        )
    )
    recipe = recipe_with(fixtures, ["top_k", "prompt_logprobs", "top_k_floor"])
    backend = build_backend(recipe, http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "prompt_logprobs"
    assert answer.degraded is False and answer.missing == ()


def test_refuses_an_answer_that_merges_with_the_prompt_tail(fixtures, fake_server_factory):
    # The template ends right at "Answer:", so the appended "no" fuses into one piece with it
    # and the last prompt row is not the answer's: reading it would read the wrong position.
    server, http = fake_server_factory(Behaviour(scores={"yes": 0.0, "no": -1.0}, **TENSORFOLD))
    recipe = recipe_with(fixtures, ["prompt_logprobs"])
    template = recipe.template.model_copy(
        update={"raw": recipe.template.raw.replace("</think>\n\n", "</think>Answer:")}
    )
    backend = build_backend(recipe.model_copy(update={"template": template}), http=http)
    with pytest.raises(BackendError, match="own prompt token"):
        run(backend.evaluate(run(backend.warm(STATE)), [Q]))


def test_a_messages_mode_recipe_cannot_pin_the_rung(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={" yes": 0.0, " no": -1.0}))
    recipe = recipe_with(
        fixtures, ["prompt_logprobs"], rung="prompt_logprobs", name="fake-vllm.yaml"
    )
    backend = build_backend(recipe, http=http)
    with pytest.raises(BackendError, match="raw-mode prompt"):
        run(backend.evaluate(run(backend.warm(STATE)), [Q]))
