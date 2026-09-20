"""The backend port over an OpenAI-compatible endpoint, built through the composition root."""

from __future__ import annotations

import asyncio
import math

import httpx
import pytest

from jevify.compose import build_backend
from jevify.domain.distribution import Distribution
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, State
from jevify.ports.backend import BackendError, ContextLimitError
from jevify.recipes import load_recipe
from tests.fakes.fake_openai_server import FILLER_TOKENS, Behaviour

STATE = State.from_jev("The customer wrote: keep my subscription, stop the newsletters.")
CHOICE = ChoiceQuestion(
    id="dept",
    instructions="Which team handles this?",
    options=(Option("billing", "invoices and refunds"), Option("support", None)),
)


def run(coro):
    return asyncio.run(coro)


def test_evaluate_choice_reads_distribution_from_top_logprobs(fixtures, fake_server_factory):
    # The fake's "model": " A" at logit 0, " B" at -1, five fillers at -3 each.
    server, http = fake_server_factory(Behaviour(scores={" A": 0.0, " B": -1.0}))
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    backend = build_backend(recipe, http=http)

    handle = run(backend.warm(STATE))
    [answer] = run(backend.evaluate(handle, [CHOICE]))

    assert answer.question_id == "dept"
    assert answer.semantics == "readout"
    assert answer.rung == "top_k"
    assert answer.degraded is False
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.labels == ("billing", "support")
    assert d.probabilities[0] == pytest.approx(0.7311, abs=1e-4)
    # Off-menu mass is the fillers' share of the softmax: 5e^-3 / (1 + e^-1 + 5e^-3).
    total = 1 + math.exp(-1) + len(FILLER_TOKENS) * math.exp(-3)
    assert answer.off_menu_mass == pytest.approx(
        len(FILLER_TOKENS) * math.exp(-3) / total, abs=1e-6
    )
    assert answer.prompt_tokens > 0
    assert answer.latency_ms >= 0.0


def test_one_request_per_question_and_state_is_the_shared_prefix(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(scores={" yes": 0.0, " no": -2.0, " A": -1.0, " B": -1.5})
    )
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    backend = build_backend(recipe, http=http)
    questions = [
        NoulQuestion(id="cancel", instructions="The customer wants to cancel."),
        CHOICE,
    ]
    handle = run(backend.warm(STATE))
    answers = run(backend.evaluate(handle, questions))

    assert [a.question_id for a in answers] == ["cancel", "dept"]
    assert len(server.requests) == 2
    user_messages = [req["messages"][-1]["content"] for req in server.requests]
    prefix = "State:\nThe customer wrote: keep my subscription, stop the newsletters.\n\n"
    assert all(content.startswith(prefix) for content in user_messages)
    assert "cancel" in user_messages[0] and "cancel" not in user_messages[1]
    assert all(req["max_tokens"] == 1 and req["logprobs"] is True for req in server.requests)
    assert all(req["chat_template_kwargs"] == {"thinking": False} for req in server.requests)


def test_never_sends_prompt_logprobs(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={" yes": 0.0, " no": -1.0}))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http)
    run(backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="q", instructions="x")]))
    assert all("prompt_logprobs" not in req for req in server.requests)


def test_context_limit_error_names_the_limit(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(max_context=40, scores={" yes": 0.0, " no": -1.0}))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http)
    long_state = State.from_jev("word " * 200)
    with pytest.raises(ContextLimitError, match="40"):
        run(
            backend.evaluate(
                run(backend.warm(long_state)), [NoulQuestion(id="q", instructions="x")]
            )
        )


def test_client_retries_429_then_succeeds(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(rate_limit_first=2, scores={" yes": 0.0, " no": -1.0})
    )
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http)
    [answer] = run(
        backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="q", instructions="x")])
    )
    assert answer.rung == "top_k"
    assert len(server.requests) == 3  # two rate-limited attempts, then success


def test_client_gives_up_after_bounded_retries(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(rate_limit_first=50, scores={" yes": 0.0}))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http)
    with pytest.raises(BackendError, match="429"):
        run(backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="q", instructions="x")]))
    assert len(server.requests) <= 5


def test_transport_errors_are_backend_errors(fixtures):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(boom), base_url="http://fake")
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http)
    with pytest.raises(BackendError, match="connection refused"):
        run(backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="q", instructions="x")]))


def test_api_key_sent_when_configured(fixtures, fake_server_factory, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_KEY", "secret-123")
    seen: list[str | None] = []
    server, _ = fake_server_factory(Behaviour(scores={" yes": 0.0, " no": -1.0}))

    async def spy(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        transport = httpx.ASGITransport(app=server.app)
        return await transport.handle_async_request(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(spy), base_url="http://fake")
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    recipe = recipe.model_copy(
        update={"endpoint": recipe.endpoint.model_copy(update={"api_key_env": "FAKE_KEY"})}
    )
    backend = build_backend(recipe, http=http)
    run(backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="q", instructions="x")]))
    assert seen == ["Bearer secret-123"]


def test_auto_rung_without_probe_fails_loudly(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={" yes": 0.0}))
    recipe = load_recipe(fixtures / "recipes" / "fake-raw.yaml")  # readout.rung: auto, no probe
    backend = build_backend(recipe, http=http)
    with pytest.raises(BackendError, match="probe"):
        run(backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="q", instructions="x")]))


def test_label_missing_from_top_k_fails_loudly_on_pinned_top_k(fixtures, fake_server_factory):
    # " no" is far below five fillers, so it never enters a top-5 list.
    server, http = fake_server_factory(Behaviour(top_k_cap=5, scores={" yes": 0.0, " no": -9.0}))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http)
    with pytest.raises(BackendError, match="false"):
        run(backend.evaluate(run(backend.warm(STATE)), [NoulQuestion(id="q", instructions="x")]))


def test_per_option_choice_composes_yes_no_readouts(fixtures, fake_server_factory):
    # The fake scores each option's document differently: billing gets yes, support gets no.
    server, http = fake_server_factory(
        Behaviour(
            dialect="llamacpp",
            scores_when=[
                ("answer is: billing", {"yes": 0.0, "no": -2.0}),
                ("answer is: support", {"yes": -2.0, "no": 0.0}),
            ],
        )
    )
    recipe = load_recipe(fixtures / "recipes" / "fake-raw.yaml")
    recipe = recipe.model_copy(
        update={"readout": recipe.readout.model_copy(update={"rung": "top_k"})}
    )
    backend = build_backend(recipe, http=http)
    question = ChoiceQuestion(
        id="dept", instructions="Which team?", options=(Option("billing"), Option("support"))
    )
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    assert answer.calls == 2
    assert len(server.requests) == 2
    d = Distribution.from_logprobs(answer.logprobs)
    # Each option's score is its yes-minus-no logit: +2 and -2, so softmax gives e^4 : 1.
    assert d.as_mapping()["billing"] == pytest.approx(math.exp(4) / (1 + math.exp(4)))
    assert answer.off_menu_mass is None
    assert answer.rung == "top_k"
