"""Suites, metrics, scorecards and `jevify eval` (ADR-0005)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from jevify.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    mean_absolute_error,
    ranked_probability_score,
)
from jevify.evaluation.suites import SuiteError, load_suite, read_rows, suite_hash

SUITES = Path(__file__).resolve().parents[1] / "suites"


def test_read_rows_rejects_invalid(tmp_path: Path) -> None:
    base = {"instruction": "Choose", "context": "State", "options": ["A", "B"], "case_id": "c1"}
    for update in ({"label": True}, {"label": 2}, {"options": ["A", "A"]}, {"instruction": ""}):
        path = tmp_path / "rows.jsonl"
        path.write_text(json.dumps({**base, **update}) + "\n")
        with pytest.raises(SuiteError):
            read_rows(path)
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n")
    with pytest.raises(SuiteError, match="case_id"):
        read_rows(path)


def test_suite_hash_matches_manifest() -> None:
    suite = load_suite(SUITES / "policy-29.jsonl")
    assert suite.manifest["sha256"] == suite_hash(SUITES / "policy-29.jsonl")
    assert (
        suite.manifest["sha256"]
        == hashlib.sha256((SUITES / "policy-29.jsonl").read_bytes()).hexdigest()
    )
    assert len(suite.rows) == 29 == suite.manifest["cases"]
    assert suite.rows[0].case_id == "original_0" and suite.rows[0].label == 1


def test_load_suite_refuses_a_tampered_file(tmp_path: Path) -> None:
    (tmp_path / "s.jsonl").write_text((SUITES / "policy-29.jsonl").read_text() + "\n")
    (tmp_path / "s.manifest.json").write_text((SUITES / "policy-29.manifest.json").read_text())
    with pytest.raises(SuiteError, match="sha256"):
        load_suite(tmp_path / "s.jsonl")


def test_metrics_from_literal_rows() -> None:
    # Two 2-way decisions: p(correct) = 0.8 then 0.4.
    probs = [{"a": 0.8, "b": 0.2}, {"a": 0.4, "b": 0.6}]
    labels = ["a", "a"]
    assert log_loss(probs, labels) == pytest.approx((-math.log(0.8) - math.log(0.4)) / 2)
    # Brier per decision: sum over labels of (p - onehot)^2: [0.04+0.04, 0.36+0.36] -> mean 0.4
    assert brier_score(probs, labels) == pytest.approx((0.08 + 0.72) / 2)
    # ECE, 10 bins: confidence 0.8 (bin 8, correct) and 0.6 (bin 6, wrong):
    # each bin has one decision: |acc - conf| = 0.2 and 0.6, weighted 0.5 each -> 0.4
    ece, bins = expected_calibration_error(probs, labels, bins=10)
    assert ece == pytest.approx(0.4)
    assert [b["count"] for b in bins] == [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    # Ordinal: expectation vs label index
    assert mean_absolute_error([1.04, 0.5], [1, 0]) == pytest.approx((0.04 + 0.5) / 2)
    # RPS over 3 levels: cdf(p) = [0.05, 0.91, 1.0], cdf(label 1) = [0, 1, 1]
    # squared diffs: 0.0025 + 0.0081 + 0 = 0.0106; divided by (k - 1) = 2 -> 0.0053
    assert ranked_probability_score([[0.05, 0.86, 0.09]], [1]) == pytest.approx(0.0053)


def test_eval_writes_runs_and_evidence(fixtures, fake_http_server_factory, tmp_path, capsys):
    from jevify.cli import main
    from tests.fakes.fake_openai_server import Behaviour

    live = fake_http_server_factory(
        Behaviour(
            scores={" A": 0.0, " B": -1.0, " C": -2.0},
            max_context=100_000,
            simulate_latency=True,
            cold_ms=20,
            warm_ms=5,
        )
    )
    doc = yaml.safe_load((fixtures / "recipes" / "fake-vllm.yaml").read_text())
    doc["endpoint"]["base_url"] = live.base_url
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(yaml.safe_dump(doc))
    runs = tmp_path / "runs"
    evidence = tmp_path / "evidence"

    code = main(
        [
            "eval",
            str(recipe_path),
            str(SUITES / "policy-29.jsonl"),
            "--runs-dir",
            str(runs),
            "--evidence-dir",
            str(evidence),
            "--concurrency",
            "4",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    raw_path = Path(report["raw"])
    assert raw_path.exists() and raw_path.is_relative_to(runs)
    lines = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    assert "header" in lines[0]  # the first line records recipe, suite, backend and versions
    raw_rows = lines[1:]
    assert len(raw_rows) == 29
    assert {"case_id", "label", "selected", "probabilities", "rung", "latency_ms"} <= set(
        raw_rows[0]
    )

    summary_json = Path(report["summary_json"])
    summary_md = Path(report["summary_md"])
    assert summary_json.is_relative_to(evidence) and summary_md.exists()
    summary = json.loads(summary_json.read_text())
    assert summary["recipe_hash"] == report["recipe_hash"]
    assert summary["suite_sha256"].startswith("c59cb29f")
    assert summary["raw_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert summary["metrics"]["cases"] == 29
    # The fake always prefers option A, so accuracy is the share of cases whose label is 0.
    first_option_is_right = sum(row["label"] == row["options"][0] for row in raw_rows)
    assert summary["metrics"]["accuracy"] == pytest.approx(first_option_is_right / 29)
    assert 0.0 <= summary["metrics"]["ece"] <= 1.0
    assert "command" in summary and "policy-29" in summary["command"]
    assert summary["backend"]["dialect"] == "vllm"
    assert "| accuracy |" in summary_md.read_text()
    assert live.server.max_in_flight >= 2  # --concurrency 4 ran cases in parallel


def test_summary_is_regenerated_from_raw(fixtures, fake_http_server_factory, tmp_path, capsys):
    """The committed evidence derives from the raw file by committed code, never typed in."""
    from jevify.cli import main
    from jevify.evaluation.scorecard import summarize_raw
    from tests.fakes.fake_openai_server import Behaviour

    live = fake_http_server_factory(Behaviour(scores={" A": 0.0, " B": -1.0}, max_context=100_000))
    doc = yaml.safe_load((fixtures / "recipes" / "fake-vllm.yaml").read_text())
    doc["endpoint"]["base_url"] = live.base_url
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(doc))
    main(
        [
            "eval",
            str(tmp_path / "r.yaml"),
            str(SUITES / "policy-29.jsonl"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    again = summarize_raw(Path(report["raw"]))
    original = json.loads(Path(report["summary_json"]).read_text())
    assert again["metrics"] == original["metrics"]
    assert again["raw_sha256"] == original["raw_sha256"]
