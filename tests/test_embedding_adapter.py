"""The embedding kind (ADR-0002 D2): cosine pertinence, labeled similarity, choice only."""

from __future__ import annotations

import asyncio

import pytest

from jevify.compose import build_backend
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, State
from jevify.ports.backend import BackendError
from jevify.recipes import load_recipe
from tests.fakes.fake_openai_server import Behaviour

STATE = State.from_jev("I was charged twice and want a refund.")
VECTORS = {
    # unit vectors in a 3-space: the state sits between billing and support, nearer billing
    "Instruct: Which team?\nQuery: I was charged twice and want a refund.": [0.8, 0.6, 0.0],
    "billing": [1.0, 0.0, 0.0],
    "support": [0.0, 1.0, 0.0],
    "sales": [0.0, 0.0, 1.0],
}


def run(coro):
    return asyncio.run(coro)


def test_embedding_adapter_scores_cosine_per_option(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(embeddings=VECTORS))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-embedding.yaml"), http=http)
    question = ChoiceQuestion(
        id="dept",
        instructions="Which team?",
        options=(Option("billing"), Option("support"), Option("sales")),
    )
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    assert answer.semantics == "similarity"
    assert answer.raw["cosine"]["billing"] == pytest.approx(0.8)
    assert answer.raw["cosine"]["support"] == pytest.approx(0.6)
    assert answer.raw["cosine"]["sales"] == pytest.approx(0.0)
    # logprobs carry cosine times the recipe's scale (10): a ranking convenience
    assert answer.logprobs["billing"] == pytest.approx(8.0)
    assert answer.calls == 2  # one call for the query, one batched call for the options
    paths = [r["path"] for r in server.requests]
    assert paths == ["/v1/embeddings", "/v1/embeddings"]
    assert server.requests[1]["input"] == ["billing", "support", "sales"]


def test_embedding_adapter_caches_option_vectors(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(embeddings=VECTORS))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-embedding.yaml"), http=http)
    question = ChoiceQuestion(
        id="q", instructions="Which team?", options=(Option("billing"), Option("sales"))
    )
    handle = run(backend.warm(STATE))
    run(backend.evaluate(handle, [question]))
    run(backend.evaluate(handle, [question]))
    option_requests = [
        r for r in server.requests if r["input"] != VECTORS and isinstance(r["input"], list)
    ]
    assert len(option_requests) == 1  # the second evaluation reused the cached option vectors


def test_embedding_kind_refuses_noul(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(embeddings=VECTORS))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-embedding.yaml"), http=http)
    with pytest.raises(BackendError, match="choice"):
        run(backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="n", instructions="x")]))


def test_embedding_probe_records_dimension(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(embeddings=VECTORS))
    caps = run(
        build_backend(load_recipe(fixtures / "recipes" / "fake-embedding.yaml"), http=http).probe()
    )
    assert caps.kind == "embedding"
    assert caps.rungs == frozenset()
    assert any("dimension 3" in note for note in caps.notes)


def test_cosine_of_unit_vectors_is_the_dot_product():
    from jevify.adapters.embedding.adapter import cosine

    assert cosine([0.8, 0.6, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(0.8)
