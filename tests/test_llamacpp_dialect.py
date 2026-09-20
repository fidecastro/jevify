"""The llama.cpp dialect: grammar and equal-bias rungs, id matching, native multimodal path."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path

import pytest

from jevify.adapters.endpoint.dialects import parse_response
from jevify.compose import build_backend
from jevify.domain.distribution import Distribution
from jevify.domain.questions import ImagePart, NoulQuestion, State, TextPart
from jevify.ports.backend import Dialect
from jevify.recipes import load_recipe
from tests.fakes.fake_openai_server import Behaviour

RESPONSES = Path(__file__).parent / "fixtures" / "responses"
STATE = State.from_jev("The customer wrote: I was charged twice and I am furious.")
Q = NoulQuestion(id="cancel", instructions="The customer threatens to cancel.")


def run(coro):
    return asyncio.run(coro)


def raw_recipe(fixtures, rung):
    recipe = load_recipe(fixtures / "recipes" / "fake-raw.yaml")
    return recipe.model_copy(update={"readout": recipe.readout.model_copy(update={"rung": rung})})


def test_grammar_rung_renormalizes_over_the_answer_set(fixtures, fake_server_factory):
    # Raw scores: yes 0, no -1, plus fillers at -3. Under a grammar the fillers are excluded and
    # post-sampling probabilities are renormalized over the survivors: yes:no = e^0 : e^-1.
    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", grammar=True, scores={"yes": 0.0, "no": -1.0})
    )
    backend = build_backend(raw_recipe(fixtures, "grammar"), http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "grammar"
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert answer.off_menu_mass is None  # the mask hides the off-menu mass
    req = server.requests[-1]
    assert req["post_sampling_probs"] is True
    assert req["grammar"].startswith("root ::=") and '"yes"' in req["grammar"]
    assert req["temperature"] == 1.0 and req["top_k"] == 0 and req["min_p"] == 0.0


def test_equal_bias_rung_keeps_ratios_exact(fixtures, fake_server_factory):
    # " no" is far below the fillers; equal bias on both answers lifts them into view.
    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", top_k_cap=3, scores={"yes": 0.0, "no": -9.0})
    )
    backend = build_backend(raw_recipe(fixtures, "equal_bias"), http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.rung == "equal_bias"
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] == pytest.approx(1 / (1 + math.exp(-9)))
    req = server.requests[-1]
    assert set(req["logit_bias"].values()) == {50}
    assert req["post_sampling_probs"] is True


def test_entries_are_matched_by_id_before_text(fixtures, fake_server_factory):
    # The fake's llama.cpp entries carry ids; a recipe token with the right id but a
    # different spelling still matches, proving id takes precedence.
    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", scores={"yes": 0.0, "no": -1.0})
    )
    recipe = raw_recipe(fixtures, "top_k")
    doc = recipe.model_dump(mode="json")
    doc["answers"]["noul"]["true"] = [{"text": "YES-misspelled", "id": 9693}]
    doc["answers"]["noul"]["false"] = [{"text": "no", "id": 2152}]
    from jevify.recipes.schema import Recipe

    backend = build_backend(Recipe.model_validate(doc), http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert set(answer.logprobs) == {"false", "true"}


def test_tokens_cached_top_level_is_cache_evidence():
    # Older llama.cpp builds report the reuse count at the top level only.
    response = json.loads((RESPONSES / "llamacpp-completions-top_k.json").read_text())
    response["usage"].pop("prompt_tokens_details", None)
    response["tokens_cached"] = 42
    parsed = parse_response("completions", response, Dialect.LLAMACPP)
    assert parsed.cached_tokens == 42
    assert parsed.entries[0].id == 9693


def test_recorded_responses_parse_like_the_fake(fixtures, fake_server_factory):
    """The fake cannot drift from reality unnoticed: real responses and fake responses go
    through the same parser and yield the same entry shape."""
    real_topk = parse_response(
        "completions",
        json.loads((RESPONSES / "llamacpp-completions-top_k.json").read_text()),
        Dialect.LLAMACPP,
    )
    real_grammar = parse_response(
        "completions",
        json.loads((RESPONSES / "llamacpp-completions-grammar.json").read_text()),
        Dialect.LLAMACPP,
    )
    real_native = parse_response(
        "completion",
        json.loads((RESPONSES / "llamacpp-native-completion.json").read_text()),
        Dialect.LLAMACPP,
    )
    real_vllm = parse_response(
        "chat/completions",
        json.loads((RESPONSES / "vllm-chat-top_k.json").read_text()),
        Dialect.VLLM,
    )
    for parsed in (real_topk, real_grammar, real_native, real_vllm):
        assert parsed.entries and parsed.prompt_tokens
        assert all(isinstance(e.logprob, float) and e.logprob <= 0.0 for e in parsed.entries)
    assert all(e.id is not None for e in real_topk.entries)
    assert real_grammar.entries[0].text == "yes"
    assert real_native.cached_tokens == 101

    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", grammar=True, scores={"yes": 0.0, "no": -1.0})
    )
    backend = build_backend(raw_recipe(fixtures, "grammar"), http=http)
    run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    fake_body = server.responses[-1]
    fake_grammar = parse_response("completions", fake_body, Dialect.LLAMACPP)
    assert {e.text for e in fake_grammar.entries} >= {"yes", "no"}
    assert all(e.id is not None for e in fake_grammar.entries)


def test_raw_multimodal_uses_native_completion_with_media_marker(fixtures, fake_server_factory):
    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", multimodal=True, scores={"yes": 0.0, "no": -1.0})
    )
    backend = build_backend(raw_recipe(fixtures, "top_k"), http=http)
    png = base64.b64encode(b"\x89PNG fake").decode()
    state = State(
        (TextPart("Screenshot: "), ImagePart(f"data:image/png;base64,{png}"), TextPart(" end"))
    )
    [answer] = run(backend.evaluate(run(backend.warm(state)), [Q]))
    req = server.requests[-1]
    assert req["path"] == "/completion"
    assert req["multimodal_data"] == [png]
    assert "<__media__>" in req["prompt"]
    assert req["prompt"].index("Screenshot") < req["prompt"].index("<__media__>")
    assert set(answer.logprobs) == {"false", "true"}


def test_grammar_rung_floors_a_label_the_server_dropped(fixtures, fake_server_factory):
    # A very certain "yes" drives "no" below float precision; llama.cpp then omits it from
    # the post-sampling list. That is a certain answer, not a broken readout.
    server, http = fake_server_factory(
        Behaviour(dialect="llamacpp", grammar=True, scores={"yes": 0.0, "no": -800.0})
    )
    backend = build_backend(raw_recipe(fixtures, "grammar"), http=http)
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [Q]))
    assert answer.degraded is False
    assert answer.missing == ("false",)
    d = Distribution.from_logprobs(answer.logprobs)
    assert d.as_mapping()["true"] > 0.999999
