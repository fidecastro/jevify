"""The Engine: warm, evaluate, and the Answer every caller receives."""

from __future__ import annotations

import asyncio
import math

import pytest

from jevify.domain.engine import Engine
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, ScoreQuestion, State
from jevify.recipes import load_recipe
from tests.fakes.fake_backend import FakeBackend

LN = math.log


def run(coro):
    return asyncio.run(coro)


def make_engine(fixtures, scripted):
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    return Engine(FakeBackend(scripted), recipe), recipe


def test_evaluate_returns_readout_label_and_rung(fixtures):
    engine, _ = make_engine(fixtures, {"dept": {"billing": LN(0.8), "support": LN(0.2)}})
    question = ChoiceQuestion(
        id="dept", instructions="Which team?", options=(Option("billing"), Option("support"))
    )
    result = run(engine.ask(State.from_jev("charged twice"), [question]))
    [answer] = result.answers
    assert answer.question_id == "dept"
    assert answer.kind == "choice"
    assert answer.semantics == "readout"
    assert answer.readout.rung == "top_k"
    assert answer.distribution.as_mapping() == pytest.approx({"billing": 0.8, "support": 0.2})
    assert answer.selected == "billing"
    assert answer.confidence == pytest.approx(answer.distribution.confidence("choice"))
    assert answer.readout.off_menu_mass == pytest.approx(0.01)
    assert result.state_id == "state-1"


def test_score_answer_carries_expectation_and_legend(fixtures):
    engine, _ = make_engine(fixtures, {"urg": {"0": LN(0.05), "1": LN(0.86), "2": LN(0.09)}})
    question = ScoreQuestion(id="urg", instructions="How urgent?", levels=("low", "mid", "high"))
    [answer] = run(engine.ask(State.from_jev("x"), [question])).answers
    assert answer.kind == "score"
    assert answer.expectation == pytest.approx(1.04)
    assert answer.legend == {"0": "low", "1": "mid", "2": "high"}
    assert answer.selected == "1"


def test_noul_answer_is_probability_of_true(fixtures):
    engine, _ = make_engine(fixtures, {"spam": {"false": LN(0.3), "true": LN(0.7)}})
    [answer] = run(
        engine.ask(State.from_jev("x"), [NoulQuestion(id="spam", instructions="It is spam.")])
    ).answers
    assert answer.kind == "noul"
    assert answer.probability_true == pytest.approx(0.7)
    assert answer.confidence == pytest.approx(0.4)


def test_usage_sums_prompt_tokens_and_one_output_token_per_question(fixtures):
    engine, _ = make_engine(
        fixtures, {"a": {"false": 0.0, "true": -1.0}, "b": {"false": -1.0, "true": 0.0}}
    )
    questions = [NoulQuestion(id="a", instructions="a"), NoulQuestion(id="b", instructions="b")]
    result = run(engine.ask(State.from_jev("x"), questions))
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 2


def test_evaluate_is_one_warm_then_one_evaluate(fixtures):
    backend = FakeBackend({"a": {"false": 0.0, "true": -1.0}})
    engine = Engine(backend, load_recipe(fixtures / "recipes" / "fake-vllm.yaml"))
    run(engine.ask(State.from_jev("x"), [NoulQuestion(id="a", instructions="a")]))
    assert len(backend.warmed) == 1
    assert backend.evaluated == [("state-1", ("a",))]
