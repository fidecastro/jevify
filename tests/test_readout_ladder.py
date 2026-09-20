"""The readout ladder as a whole: auto-selection, the floor rung, and pinned-but-unproven."""

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


def run(coro):
    return asyncio.run(coro)


def with_probe(recipe, rungs, rung="auto"):
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


def test_auto_selects_strongest_proven_rung(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={" yes": 0.0, " no": -1.0}))
    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"),
        ["top_k", "top_k_floor", "named_token_logprobs"],
    )
    backend = build_backend(recipe, http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "named_token_logprobs"
    assert "logprob_token_ids" in server.requests[-1]

    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), ["top_k", "top_k_floor"]
    )
    [answer] = run(build_backend(recipe, http=http).evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "top_k"


def test_named_token_logprobs_exact_beyond_top_k(fixtures, fake_server_factory):
    # The server lists only one token, yet named ids recover both answers exactly.
    server, http = fake_server_factory(Behaviour(top_k_cap=1, scores={" yes": 0.0, " no": -1.0}))
    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"),
        ["named_token_logprobs", "top_k"],
        rung="named_token_logprobs",
    )
    backend = build_backend(recipe, http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert answer.off_menu_mass is not None and answer.off_menu_mass > 0
    assert answer.degraded is False


def test_floor_marks_degraded_and_lists_missing(fixtures, fake_server_factory):
    # " no" never reaches the top-5, so the floor rung assigns it the observed floor minus a
    # margin, marks the answer degraded, and names the missing label.
    server, http = fake_server_factory(Behaviour(top_k_cap=5, scores={" yes": 0.0, " no": -9.0}))
    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"),
        ["top_k", "top_k_floor"],
        rung="top_k_floor",
    )
    backend = build_backend(recipe, http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "top_k_floor"
    assert answer.degraded is True
    assert answer.missing == ("false",)
    observed_floor = min(e["logprob"] for e in answer.raw["parts"][0]["entries"])
    assert answer.logprobs["false"] < observed_floor
    # yes at logit 0 over {yes, no at -9, five fillers at -3}: its raw logprob is exact.
    expected = math.log(1 / (1 + math.exp(-9) + 5 * math.exp(-3)))
    assert answer.logprobs["true"] == pytest.approx(expected, abs=1e-6)
    assert answer.off_menu_mass is None


def test_pinned_unproven_rung_fails_loudly(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={" yes": 0.0, " no": -1.0}))
    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"),
        ["top_k", "top_k_floor"],
        rung="named_token_logprobs",
    )
    backend = build_backend(recipe, http=http)
    with pytest.raises(BackendError, match="did not prove"):
        run(backend.evaluate(run(backend.warm(STATE)), [Q]))


def test_auto_walks_down_the_ladder_when_a_label_is_outside_the_top_k(
    fixtures, fake_server_factory
):
    """Auto is a ladder, not a pick: when the strongest proven rung cannot read every label
    it hands the question to the next proven rung, and the answer names the rung that read
    it. With only the floor left, the answer is floored and marked degraded."""
    # " no" sits far below five fillers, so it never reaches the top-5 the recipe reads.
    server, http = fake_server_factory(
        Behaviour(
            logprobs_mode="processed",
            top_k_cap=5,
            scores={" yes": 0.0, " no": -9.0, "yes": -0.5, "no": -8.0},
        )
    )
    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"),
        ["top_k", "equal_bias", "top_k_floor"],
    )
    backend = build_backend(recipe, http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "equal_bias"
    assert answer.degraded is False and answer.missing == ()
    # the same question with only the floor below top_k: floored, degraded, missing named
    server, http = fake_server_factory(Behaviour(top_k_cap=5, scores={" yes": 0.0, " no": -9.0}))
    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), ["top_k", "top_k_floor"]
    )
    backend = build_backend(recipe, http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "top_k_floor" and answer.degraded is True
    assert answer.missing == ("false",)
    # a pinned rung never walks: it fails loudly, as before
    recipe = with_probe(
        load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), ["top_k", "top_k_floor"], rung="top_k"
    )
    backend = build_backend(recipe, http=http)
    with pytest.raises(BackendError, match="not in the top-"):
        run(backend.evaluate(run(backend.warm(STATE)), [Q]))
