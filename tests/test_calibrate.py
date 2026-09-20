"""Calibration (ADR-0005 D5): post-hoc temperature per bucket, off by default, evidenced."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import yaml

from jevify.domain.distribution import Distribution
from jevify.domain.engine import Engine
from jevify.domain.questions import ChoiceQuestion, Option, State
from jevify.evaluation.calibrate import bucket_key, fit_temperature, scale_logprobs
from jevify.recipes import load_recipe
from tests.fakes.fake_backend import FakeBackend

SUITES = Path(__file__).resolve().parents[1] / "suites"


def test_bucket_keys_follow_laya_option_bands():
    assert bucket_key("choice", 2) == "choice:2"
    assert bucket_key("choice", 4) == "choice:3-5"
    assert bucket_key("score", 8) == "score:6-10"
    assert bucket_key("choice", 30) == "choice:11+"
    assert bucket_key("noul", 2) == "noul:2"


def test_temperature_scaling_never_changes_the_ranking():
    logprobs = {"a": -0.1, "b": -2.0, "c": -4.0}
    for temperature in (0.3, 1.0, 4.0):
        scaled = scale_logprobs(logprobs, temperature)
        assert Distribution.from_logprobs(scaled).argmax() == "a"
        ranking = sorted(scaled, key=scaled.get)
        assert ranking == ["c", "b", "a"]


def test_fit_recovers_a_known_overconfidence():
    # Ground truth: outcomes drawn from p. Reported logits are the true logits times 2
    # (overconfident). The fitted temperature should be close to 2.
    rng = random.Random(7)
    rows = []
    for _ in range(600):
        p = rng.uniform(0.55, 0.95)
        true_logits = {"a": math.log(p), "b": math.log(1 - p)}
        reported = {k: 2.0 * v for k, v in true_logits.items()}
        label = "a" if rng.random() < p else "b"
        rows.append((reported, label))
    temperature = fit_temperature(rows)
    assert 1.6 < temperature < 2.4


def test_calibrate_writes_table_and_evidence(fixtures, fake_http_server_factory, tmp_path, capsys):
    from jevify.cli import main
    from tests.fakes.fake_openai_server import Behaviour

    live = fake_http_server_factory(
        Behaviour(scores={" A": 0.0, " B": -0.6, " C": -1.2}, max_context=100_000)
    )
    doc = yaml.safe_load((fixtures / "recipes" / "fake-vllm.yaml").read_text())
    doc["endpoint"]["base_url"] = live.base_url
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(yaml.safe_dump(doc))
    main(
        [
            "eval",
            str(recipe_path),
            str(SUITES / "policy-29.jsonl"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )
    raw = json.loads(capsys.readouterr().out)["raw"]

    code = main(
        [
            "calibrate",
            str(recipe_path),
            raw,
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--min-fit",
            "3",
            "--force",  # the 29-case fake run has error-free buckets; this test is about writing
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    updated = load_recipe(recipe_path)
    assert updated.calibration is not None
    assert set(updated.calibration.temperatures) == {"choice:2", "choice:3-5"}
    assert updated.calibration.temperatures["choice:3-5"] > 0
    assert updated.calibration.evidence and Path(updated.calibration.evidence).exists()
    evidence = json.loads(Path(report["evidence_json"]).read_text())
    assert evidence["split"]["fit"] + evidence["split"]["held_out"] == 29
    assert {"log_loss", "brier", "ece"} <= set(evidence["held_out_before"])
    assert {"log_loss", "brier", "ece"} <= set(evidence["held_out_after"])
    assert "reliability_after" in evidence
    assert report["recipe_hash_after"] != report["recipe_hash_before"]


def test_engine_applies_the_table_and_labels_calibrated(fixtures):
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    recipe = recipe.model_copy(
        update={"calibration": {"temperatures": {"choice:2": 2.0}, "evidence": "x"}}
    )
    from jevify.recipes.schema import Recipe

    recipe = Recipe.model_validate(recipe.model_dump(mode="json"))
    backend = FakeBackend({"q": {"a": 0.0, "b": -2.0}})
    engine = Engine(backend, recipe)
    q = ChoiceQuestion(id="q", instructions="Pick.", options=(Option("a"), Option("b")))
    [answer] = __import__("asyncio").run(engine.ask(State.from_jev("x"), [q])).answers
    assert answer.semantics == "calibrated"
    assert answer.readout.temperature == 2.0
    # logit gap 2.0 scaled by 1/2 -> gap 1.0 -> p(a) = 1 / (1 + e^-1)
    assert answer.distribution.as_mapping()["a"] == pytest.approx(1 / (1 + math.exp(-1)))
    # a bucket the table does not cover stays a plain readout
    q3 = ChoiceQuestion(
        id="q3", instructions="Pick.", options=(Option("a"), Option("b"), Option("c"))
    )
    backend.scripted["q3"] = {"a": 0.0, "b": -1.0, "c": -2.0}
    [answer3] = __import__("asyncio").run(engine.ask(State.from_jev("x"), [q3])).answers
    assert answer3.semantics == "readout" and answer3.readout.temperature is None


def test_calibrate_refuses_a_table_it_cannot_defend(
    fixtures, fake_http_server_factory, tmp_path, capsys
):
    """Too few decisions, or no error to learn from: the table is reported, not applied."""
    from jevify.cli import main
    from tests.fakes.fake_openai_server import Behaviour

    live = fake_http_server_factory(
        Behaviour(scores={" A": 0.0, " B": -0.6, " C": -1.2}, max_context=100_000)
    )
    doc = yaml.safe_load((fixtures / "recipes" / "fake-vllm.yaml").read_text())
    doc["endpoint"]["base_url"] = live.base_url
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(yaml.safe_dump(doc))
    main(
        [
            "eval",
            str(recipe_path),
            str(SUITES / "policy-29.jsonl"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )
    raw = json.loads(capsys.readouterr().out)["raw"]
    before = load_recipe(recipe_path)
    code = main(["calibrate", str(recipe_path), raw, "--evidence-dir", str(tmp_path / "evidence")])
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] is False
    assert any("fewer than 50" in p for p in report["problems"])
    assert load_recipe(recipe_path).calibration is None
    assert load_recipe(recipe_path).model_dump() == before.model_dump()
    evidence = json.loads(Path(report["evidence_json"]).read_text())
    assert evidence["applicable"] is False
    forced = main(
        [
            "calibrate",
            str(recipe_path),
            raw,
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--force",
        ]
    )
    assert forced == 0
    assert json.loads(capsys.readouterr().out)["forced"] is True
