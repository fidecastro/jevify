"""The encoder kind (ADR-0002 D2): an NLI head read at each option, labeled readout."""

from __future__ import annotations

import asyncio
import math

import pytest

from jevify.adapters.encoder.adapter import EncoderBackend
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, ScoreQuestion, State
from jevify.ports.backend import BackendError
from jevify.recipes import load_recipe

STATE = State.from_jev("How long should I boil an egg?")


class FakeScorer:
    """Logits by hypothesis text; any hypothesis it does not know scores neutral."""

    def __init__(
        self, table: dict[str, dict[str, float]], labels=("contradiction", "entailment", "neutral")
    ):
        self.table = table
        self.labels = labels
        self.calls: list[list[tuple[str, str]]] = []

    async def score(self, pairs):
        self.calls.append(list(pairs))
        return [self.table.get(h, {label: 0.0 for label in self.labels}) for _, h in pairs]

    def describe(self):
        return {"transport": "fake", "dialect": None}


def run(coro):
    return asyncio.run(coro)


def make(fixtures, table):
    return EncoderBackend(
        FakeScorer(table), load_recipe(fixtures / "recipes" / "fake-encoder.yaml")
    )


def test_choice_reads_entailment_minus_contradiction_per_option(fixtures):
    backend = make(
        fixtures,
        {
            "This text is about cooking.": {
                "contradiction": 0.0,
                "entailment": 2.0,
                "neutral": 1.0,
            },
            "This text is about astronomy.": {
                "contradiction": 3.0,
                "entailment": -1.0,
                "neutral": 0.0,
            },
        },
    )
    question = ChoiceQuestion(
        id="q", instructions="Topic?", options=(Option("cooking"), Option("astronomy"))
    )
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    assert answer.semantics == "readout"
    assert answer.logprobs == {"cooking": 2.0, "astronomy": -4.0}
    assert answer.calls == 1
    assert backend.scorer.calls == [
        [
            ("How long should I boil an egg?", "This text is about cooking."),
            ("How long should I boil an egg?", "This text is about astronomy."),
        ]
    ]


def test_noul_is_the_two_way_distribution_at_the_statement(fixtures):
    backend = make(
        fixtures,
        {"The text is about food.": {"contradiction": 0.0, "entailment": 2.0, "neutral": 5.0}},
    )
    question = NoulQuestion(id="n", instructions="The text is about food.")
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    # neutral mass is ignored: p(true) = sigmoid(2 - 0) = 0.8808
    assert answer.logprobs["true"] == pytest.approx(math.log(0.8808), abs=1e-4)
    assert answer.logprobs["false"] == pytest.approx(math.log(0.1192), abs=1e-4)


def test_score_levels_become_hypotheses_in_order(fixtures):
    backend = make(
        fixtures,
        {
            "This text is about low.": {"contradiction": 1.0, "entailment": 0.0, "neutral": 0.0},
            "This text is about high.": {"contradiction": 0.0, "entailment": 1.5, "neutral": 0.0},
        },
    )
    question = ScoreQuestion(id="s", instructions="Urgency?", levels=("low", "high"))
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    assert answer.logprobs == {"0": -1.0, "1": 1.5}
    assert answer.raw["hypotheses"] == {
        "0": "This text is about low.",
        "1": "This text is about high.",
    }


def test_probe_names_missing_head_labels(fixtures):
    backend = EncoderBackend(
        FakeScorer({}, labels=("yes", "no")),
        load_recipe(fixtures / "recipes" / "fake-encoder.yaml"),
    )
    with pytest.raises(BackendError, match="entailment_label"):
        run(backend.probe())


def test_probe_records_layout_and_labels(fixtures):
    caps = run(make(fixtures, {}).probe())
    assert caps.kind == "encoder"
    assert any("layout nli" in note and "entailment" in note for note in caps.notes)


def test_encoder_refuses_images(fixtures):
    from jevify.domain.questions import ImagePart, TextPart

    png = "data:image/png;base64,iVBORw0KGgo="
    state = State(parts=(TextPart("x"), ImagePart(png)))
    with pytest.raises(BackendError, match="image"):
        run(make(fixtures, {}).warm(state))
