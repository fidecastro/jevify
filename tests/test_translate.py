"""Domain answers to Jev's wire shape (ADR-0003 D1, D3): the one place the shape is produced."""

from __future__ import annotations

import asyncio
import math

from jevify.api.translate import evaluation_to_jev
from jevify.domain.engine import Engine
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, ScoreQuestion, State
from jevify.recipes import load_recipe
from tests.fakes.fake_backend import FakeBackend

LN = math.log


def test_evaluation_to_jev_matches_the_sdk_shapes(fixtures):
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    backend = FakeBackend(
        {
            "dept": {"billing": LN(0.84), "support": LN(0.16)},
            "urg": {"0": LN(0.05), "1": LN(0.86), "2": LN(0.09)},
            "spam": {"false": LN(0.999), "true": LN(0.001)},
        }
    )
    engine = Engine(backend, recipe)
    questions = [
        ChoiceQuestion(
            id="dept", instructions="Team?", options=(Option("billing"), Option("support"))
        ),
        ScoreQuestion(id="urg", instructions="Urgency?", levels=("calm", "civil", "angry")),
        NoulQuestion(id="spam", instructions="Spam."),
    ]
    evaluation = asyncio.run(engine.ask(State.from_jev("x"), questions))
    body = evaluation_to_jev(evaluation, model="fake-decoder", recipe_hash="abc")

    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "fake-decoder"
    dept, urg, spam = body["answers"]["dept"], body["answers"]["urg"], body["answers"]["spam"]

    assert dept["type"] == "choice" and dept["choice"] == "billing"
    assert isinstance(dept["confidence"], float)
    assert round(dept["probabilities"]["billing"], 2) == 0.84
    assert dept["x_jevify"]["semantics"] == "readout"
    assert dept["x_jevify"]["rung"] == "top_k"
    assert dept["x_jevify"]["recipe_hash"] == "abc"

    assert urg["type"] == "score" and round(urg["score"], 2) == 1.04
    assert urg["legend"] == {"0": "calm", "1": "civil", "2": "angry"}
    assert set(urg["probabilities"]) == {"0", "1", "2"}

    assert spam["type"] == "noul" and round(spam["noul"], 3) == 0.001
    assert "confidence" not in spam  # Jev's noul answer has no confidence field
    assert "confidence" in spam["x_jevify"]

    assert body["usage"] == {"input_tokens": 300, "output_tokens": 3}
    assert all(isinstance(v, int) for v in body["usage"].values())
