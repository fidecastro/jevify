"""The rerank kind (ADR-0002 D2): one independent relevance score per option, labeled so."""

from __future__ import annotations

import asyncio
import math

import pytest

from jevify.compose import build_backend
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, State
from jevify.ports.backend import BackendError
from jevify.recipes import load_recipe
from tests.fakes.fake_openai_server import Behaviour

STATE = State.from_jev("The customer wrote: I was charged twice and I am furious.")


def run(coro):
    return asyncio.run(coro)


def test_rerank_adapter_scores_each_option_independently(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(
            dialect="llamacpp",
            rerank_scores={"The customer threatens to cancel.": 0.91, "pleased": 0.07},
        )
    )
    recipe = load_recipe(fixtures / "recipes" / "fake-rerank.yaml")
    backend = build_backend(recipe, http=http)
    question = ChoiceQuestion(
        id="q",
        instructions="Which statement holds?",
        options=(Option("threatens", "The customer threatens to cancel."), Option("pleased")),
    )
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    assert answer.semantics == "relevance"
    assert answer.raw["scores"] == {"threatens": 0.91, "pleased": 0.07}
    # logprobs carry the independent scores as log-values; the engine's softmax over them is
    # a ranking convenience, not a probability, and the label says so.
    assert answer.logprobs["threatens"] == pytest.approx(math.log(0.91))
    req = server.requests[-1]
    assert req["path"] == "/v1/rerank"
    assert req["query"].startswith("The customer wrote")
    assert req["documents"] == ["The customer threatens to cancel.", "pleased"]


def test_rerank_noul_scores_the_statement(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", rerank_scores={"The customer threatens to cancel.": 0.8})
    )
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-rerank.yaml"), http=http)
    [answer] = run(
        backend.evaluate(
            run(backend.warm(STATE)),
            [NoulQuestion(id="n", instructions="The customer threatens to cancel.")],
        )
    )
    assert answer.semantics == "relevance"
    assert answer.raw["scores"]["true"] == pytest.approx(0.8)
    assert answer.logprobs["true"] == pytest.approx(math.log(0.8))
    assert answer.logprobs["false"] == pytest.approx(math.log(0.2))


def test_rerank_probe_records_kind_and_rejects_images(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(dialect="llamacpp", rerank_scores={"x": 0.5}))
    recipe = load_recipe(fixtures / "recipes" / "fake-rerank.yaml")
    caps = run(build_backend(recipe, http=http).probe())
    assert caps.kind == "rerank"
    assert caps.modalities == {"text"}
    assert caps.rungs == frozenset()


def test_rerank_kind_refuses_score_questions(fixtures, fake_server_factory):
    from jevify.domain.questions import ScoreQuestion

    server, http = fake_server_factory(Behaviour(dialect="llamacpp", rerank_scores={"x": 0.5}))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-rerank.yaml"), http=http)
    with pytest.raises(BackendError, match="score"):
        run(
            backend.evaluate(
                run(backend.warm(STATE)),
                [ScoreQuestion(id="s", instructions="How urgent?", levels=("low", "high"))],
            )
        )
