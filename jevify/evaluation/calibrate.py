"""Post-hoc calibration (ADR-0005 D5): one temperature per question type and option-count
bucket, fitted on labeled decisions, standard library only.

A temperature rescales log-probabilities before the softmax. It never changes a ranking; it
changes how sharp the distribution is. Fitting minimizes log loss on a fit split; the
evidence reports the held-out split before and after, so the table is judged on decisions
it never saw.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jevify.evaluation.metrics import brier_score, expected_calibration_error, log_loss

Row = tuple[Mapping[str, float], str]  # (logprobs over labels, correct label)


def bucket_key(question_type: str, options: int) -> str:
    if options <= 2:
        band = "2"
    elif options <= 5:
        band = "3-5"
    elif options <= 10:
        band = "6-10"
    else:
        band = "11+"
    return f"{question_type}:{band}"


def scale_logprobs(logprobs: Mapping[str, float], temperature: float) -> dict[str, float]:
    return {label: value / temperature for label, value in logprobs.items()}


def _softmax(logprobs: Mapping[str, float]) -> dict[str, float]:
    peak = max(logprobs.values())
    weights = {k: math.exp(v - peak) for k, v in logprobs.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def _loss(rows: Sequence[Row], temperature: float) -> float:
    return log_loss(
        [_softmax(scale_logprobs(lp, temperature)) for lp, _ in rows], [y for _, y in rows]
    )


def fit_temperature(rows: Sequence[Row], *, low: float = 0.05, high: float = 20.0) -> float:
    """Golden-section search over log-temperature for the minimum log loss."""
    if not rows:
        return 1.0
    a, b = math.log(low), math.log(high)
    phi = (math.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = _loss(rows, math.exp(c)), _loss(rows, math.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _loss(rows, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _loss(rows, math.exp(d))
    return math.exp((a + b) / 2)


def _rows_from_raw(raw_path: Path) -> list[dict[str, Any]]:
    lines = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in lines[1:] if row.get("label") is not None and not row.get("error")]


def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic half split by case id hash, so a re-run reproduces the same split."""
    fit, held = [], []
    for row in rows:
        digest = hashlib.sha256(row["case_id"].encode("utf-8")).digest()[0]
        (fit if digest % 2 == 0 else held).append(row)
    return fit, held


def _as_rows(decisions: list[dict[str, Any]]) -> list[Row]:
    return [
        ({k: math.log(max(v, 1e-12)) for k, v in d["probabilities"].items()}, d["label"])
        for d in decisions
    ]


def _metrics(decisions: list[dict[str, Any]], temperatures: Mapping[str, float]) -> dict[str, Any]:
    probs, labels = [], []
    for d in decisions:
        key = bucket_key(d["question_type"], len(d["options"]))
        lp = {k: math.log(max(v, 1e-12)) for k, v in d["probabilities"].items()}
        probs.append(_softmax(scale_logprobs(lp, temperatures.get(key, 1.0))))
        labels.append(d["label"])
    if not labels:
        return {}
    ece, table = expected_calibration_error(probs, labels, bins=10)
    return {
        "decisions": len(labels),
        "log_loss": log_loss(probs, labels),
        "brier": brier_score(probs, labels),
        "ece": ece,
        "reliability": table,
    }


def bucket_problems(buckets: Mapping[str, list[dict[str, Any]]], *, min_fit: int) -> list[str]:
    """Why a fitted table must not be applied: too few decisions, or no error to learn from
    (a fit without errors drives the temperature to the bound and claims certainty)."""
    problems = []
    for key, rows in buckets.items():
        if len(rows) < min_fit:
            problems.append(f"{key}: {len(rows)} fit decisions, fewer than {min_fit}")
        if not any(max(d["probabilities"], key=d["probabilities"].get) != d["label"] for d in rows):
            problems.append(f"{key}: no wrong decision in the fit split, nothing to calibrate")
    return problems


def calibrate_run(raw_path: Path, *, recipe_hash_before: str, min_fit: int = 50) -> dict[str, Any]:
    decisions = _rows_from_raw(raw_path)
    fit, held = _split(decisions)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for d in fit:
        buckets.setdefault(bucket_key(d["question_type"], len(d["options"])), []).append(d)
    temperatures = {key: fit_temperature(_as_rows(rows)) for key, rows in buckets.items()}
    before = _metrics(held, {})
    after = _metrics(held, temperatures)
    problems = bucket_problems(buckets, min_fit=min_fit)
    return {
        "applicable": not problems,
        "problems": problems,
        "raw_file": raw_path.name,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "recipe_hash_before": recipe_hash_before,
        "split": {"fit": len(fit), "held_out": len(held), "rule": "sha256(case_id)[0] parity"},
        "temperatures": temperatures,
        "bucket_sizes": {k: len(v) for k, v in buckets.items()},
        "held_out_before": {k: v for k, v in before.items() if k != "reliability"},
        "held_out_after": {k: v for k, v in after.items() if k != "reliability"},
        "reliability_before": before.get("reliability", []),
        "reliability_after": after.get("reliability", []),
        "note": (
            "a temperature never changes a ranking; judge it on held-out log loss and ECE, and "
            "on enough decisions: below a few hundred these numbers are indicative only"
        ),
    }


def render_markdown(evidence: dict[str, Any], recipe_stem: str) -> str:
    lines = [
        f"# Calibration: {recipe_stem}",
        "",
        f"Fitted on `runs/{evidence['raw_file']}` (sha256 `{evidence['raw_sha256'][:16]}…`), "
        f"recipe hash before `{evidence['recipe_hash_before'][:16]}…`.",
        "",
        f"Split: {evidence['split']['fit']} decisions to fit, "
        f"{evidence['split']['held_out']} held out ({evidence['split']['rule']}).",
        "",
        "| bucket | fit decisions | temperature |",
        "|---|---|---|",
    ]
    for key, temperature in evidence["temperatures"].items():
        lines.append(f"| {key} | {evidence['bucket_sizes'][key]} | {temperature:.3f} |")
    lines += ["", "| held-out metric | before | after |", "|---|---|---|"]
    for key in ("log_loss", "brier", "ece"):
        before = evidence["held_out_before"].get(key)
        after = evidence["held_out_after"].get(key)
        if before is not None and after is not None:
            lines.append(f"| {key} | {before:.4f} | {after:.4f} |")
    if not evidence.get("applicable", True):
        lines += ["", "**Not applicable as fitted:**"] + [f"- {p}" for p in evidence["problems"]]
        if evidence.get("forced"):
            lines.append("- the table was written anyway with --force; treat its label with care")
    lines += ["", evidence["note"] + ".", ""]
    return "\n".join(lines)
