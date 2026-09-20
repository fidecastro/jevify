"""Warm and fan-out (ADR-0002 D6): encode the state once, then branch questions off it."""

from __future__ import annotations

import asyncio

from jevify.compose import build_backend
from jevify.domain.questions import NoulQuestion, State
from jevify.ports.backend import Capabilities, Dialect, FanoutEvidence, Rung
from jevify.recipes import load_recipe
from tests.fakes.fake_openai_server import Behaviour

LONG_STATE = State.from_jev("The customer wrote a very long complaint. " * 60)
QUESTIONS = [
    NoulQuestion(id=f"q{i}", instructions=f"Statement number {i} holds.") for i in range(4)
]
SCORES = {" yes": 0.0, " no": -1.0, "yes": 0.0, "no": -1.0}


def run(coro):
    return asyncio.run(coro)


def probed(recipe, *, concurrency=1, slots=None):
    fanout = FanoutEvidence(concurrency=concurrency, concurrent_ms=10.0, sequential_ms=40.0)
    caps = Capabilities(
        kind="endpoint",
        model=recipe.model.name,
        dialect=Dialect(recipe.endpoint.dialect),
        rungs=frozenset({Rung.TOP_K}),
        fanout=fanout,
        slots=slots,
    )
    return recipe.model_copy(
        update={
            "probe": caps.to_dict(),
            "readout": recipe.readout.model_copy(update={"rung": "top_k"}),
        }
    )


def test_warm_sends_prefix_only_with_one_token(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(scores=SCORES, max_context=100_000, block_tokens=16)
    )
    backend = build_backend(probed(load_recipe(fixtures / "recipes" / "fake-vllm.yaml")), http=http)
    handle = run(backend.warm(LONG_STATE))
    assert len(server.requests) == 1
    req = server.requests[0]
    assert req["max_tokens"] == 1
    assert req["messages"][-1]["content"].endswith("complaint. \n\n")  # state and anchor only
    assert "Question" not in req["messages"][-1]["content"]
    assert handle.warm_ms is not None and handle.warm_ms >= 0.0


def test_evaluate_after_warm_reports_cache_evidence(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(scores=SCORES, max_context=100_000, block_tokens=16)
    )
    backend = build_backend(probed(load_recipe(fixtures / "recipes" / "fake-vllm.yaml")), http=http)
    handle = run(backend.warm(LONG_STATE))
    [answer] = run(backend.evaluate(handle, QUESTIONS[:1]))
    assert answer.cached_tokens is not None and answer.cached_tokens > 0
    assert answer.cached_tokens % 16 == 0


def test_evaluate_runs_questions_concurrently_when_the_probe_says_so(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(scores=SCORES, max_context=100_000, simulate_latency=True, cold_ms=20, warm_ms=5)
    )
    recipe = probed(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), concurrency=4)
    backend = build_backend(recipe, http=http)
    handle = run(backend.warm(LONG_STATE))
    answers = run(backend.evaluate(handle, QUESTIONS))
    assert [a.question_id for a in answers] == ["q0", "q1", "q2", "q3"]
    assert server.max_in_flight == 4


def test_evaluate_is_sequential_when_the_probe_found_no_gain(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(scores=SCORES, max_context=100_000, simulate_latency=True, cold_ms=20, warm_ms=5)
    )
    recipe = probed(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), concurrency=1)
    backend = build_backend(recipe, http=http)
    handle = run(backend.warm(LONG_STATE))
    run(backend.evaluate(handle, QUESTIONS))
    assert server.max_in_flight == 1


def test_llamacpp_warms_each_slot_and_pins_questions_to_slots(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", scores=SCORES, max_context=100_000, block_tokens=16, slots=2)
    )
    recipe = probed(load_recipe(fixtures / "recipes" / "fake-raw.yaml"), concurrency=2, slots=2)
    backend = build_backend(recipe, http=http)
    handle = run(backend.warm(LONG_STATE))
    warm_requests = [r for r in server.requests]
    assert sorted(r["id_slot"] for r in warm_requests) == [0, 1]
    answers = run(backend.evaluate(handle, QUESTIONS))
    question_requests = server.requests[len(warm_requests) :]
    assert {r["id_slot"] for r in question_requests} == {0, 1}
    assert all(a.cached_tokens and a.cached_tokens > 0 for a in answers)


def test_probe_measures_fanout(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(scores=SCORES, max_context=100_000, simulate_latency=True, cold_ms=30, warm_ms=5)
    )
    caps = run(
        build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http).probe()
    )
    assert caps.fanout is not None
    assert caps.fanout.concurrency > 1
    assert caps.fanout.concurrent_ms < caps.fanout.sequential_ms
