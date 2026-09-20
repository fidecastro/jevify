"""Scorecards (ADR-0005 D2): raw per-decision output to `runs/`, a summary derived from it
by this code to `docs/evidence/`, with every hash and version a reader needs."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jevify import __version__
from jevify.domain.engine import Engine
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, ScoreQuestion, State
from jevify.evaluation.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    mean_absolute_error,
    quantile,
    ranked_probability_score,
)
from jevify.evaluation.suites import Suite, SuiteRow
from jevify.recipes.schema import Recipe
from jevify.recipes.store import recipe_hash


@dataclass(frozen=True)
class Decision:
    case_id: str
    group: str
    question_type: str
    options: list[str]
    label: str | None
    selected: str
    probabilities: dict[str, float]
    confidence: float
    semantics: str
    rung: str
    degraded: bool
    off_menu_mass: float | None
    calls: int
    warm_ms: float | None
    latency_ms: float
    prompt_tokens: int | None
    cached_tokens: int | None
    error: str | None = None


def _question(row: SuiteRow):
    if row.question_type == "noul":
        return NoulQuestion(id=row.case_id, instructions=row.instruction)
    if row.question_type == "score":
        return ScoreQuestion(id=row.case_id, instructions=row.instruction, levels=row.options)
    return ChoiceQuestion(
        id=row.case_id, instructions=row.instruction, options=tuple(Option(o) for o in row.options)
    )


def _label_key(row: SuiteRow) -> str | None:
    if row.label is None:
        return None
    if row.question_type == "choice":
        return row.options[row.label]
    if row.question_type == "noul":
        return "true" if row.label == 1 else "false"
    return str(row.label)


async def run_case(engine: Engine, row: SuiteRow) -> Decision:
    question = _question(row)
    try:
        evaluation = await engine.ask(State.from_jev(row.context), [question])
    except Exception as exc:  # noqa: BLE001 - a failed case is a recorded decision, not a crash
        return Decision(
            case_id=row.case_id,
            group=row.group,
            question_type=row.question_type,
            options=list(row.options),
            label=_label_key(row),
            selected="",
            probabilities={},
            confidence=0.0,
            semantics="readout",
            rung="",
            degraded=True,
            off_menu_mass=None,
            calls=0,
            warm_ms=None,
            latency_ms=0.0,
            prompt_tokens=None,
            cached_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    [answer] = evaluation.answers
    return Decision(
        case_id=row.case_id,
        group=row.group,
        question_type=row.question_type,
        options=list(row.options),
        label=_label_key(row),
        selected=answer.selected,
        probabilities=answer.distribution.as_mapping(),
        confidence=answer.confidence,
        semantics=answer.semantics,
        rung=answer.readout.rung,
        degraded=answer.readout.degraded,
        off_menu_mass=answer.readout.off_menu_mass,
        calls=answer.readout.calls,
        warm_ms=evaluation.warm_ms,
        latency_ms=answer.readout.latency_ms,
        prompt_tokens=answer.readout.prompt_tokens,
        cached_tokens=answer.readout.cached_tokens,
    )


async def run_suite(engine: Engine, suite: Suite, *, concurrency: int = 1) -> list[Decision]:
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(row: SuiteRow) -> Decision:
        async with gate:
            return await run_case(engine, row)

    return list(await asyncio.gather(*(one(row) for row in suite.rows)))


def write_raw(
    decisions: list[Decision], runs_dir: Path, recipe: Recipe, suite: Suite, header: dict[str, Any]
) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    target = runs_dir / f"{Path(header['recipe']).stem}--{suite.name}--{stamp}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"header": header}, ensure_ascii=False) + "\n")
        for decision in decisions:
            stream.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
    return target


def summarize_raw(raw_path: Path) -> dict[str, Any]:
    """Everything in the committed summary comes from the raw file, by this function."""
    text = raw_path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    header = lines[0]["header"]
    rows = [Decision(**row) for row in lines[1:]]
    scored = [r for r in rows if r.label is not None and not r.error]
    metrics: dict[str, Any] = {
        "cases": len(rows),
        "scored": len(scored),
        "errors": sum(1 for r in rows if r.error),
    }
    reliability: list[dict[str, float | int]] = []
    if scored:
        labels = [r.label or "" for r in scored]
        metrics["accuracy"] = accuracy([r.selected for r in scored], labels)
        metrics["log_loss"] = log_loss([r.probabilities for r in scored], labels)
        metrics["brier"] = brier_score([r.probabilities for r in scored], labels)
        metrics["ece"], reliability = expected_calibration_error(
            [r.probabilities for r in scored], labels, bins=10
        )
        ordinal = [r for r in scored if r.question_type == "score"]
        if ordinal:
            expectations = [
                sum(i * r.probabilities[str(i)] for i in range(len(r.options))) for r in ordinal
            ]
            metrics["score_mae"] = mean_absolute_error(
                expectations, [int(r.label or 0) for r in ordinal]
            )
            metrics["score_rps"] = ranked_probability_score(
                [[r.probabilities[str(i)] for i in range(len(r.options))] for r in ordinal],
                [int(r.label or 0) for r in ordinal],
            )
        groups: dict[str, dict[str, float | int]] = {}
        for group in sorted({r.group for r in scored}):
            members = [r for r in scored if r.group == group]
            groups[group] = {
                "cases": len(members),
                "accuracy": accuracy(
                    [r.selected for r in members], [r.label or "" for r in members]
                ),
            }
        metrics["groups"] = groups
        metrics["degraded"] = sum(1 for r in scored if r.degraded)
        latencies = [r.latency_ms for r in scored]
        metrics["latency_ms"] = {
            "median": statistics.median(latencies),
            "p95": quantile(latencies, 0.95),
            "mean": statistics.fmean(latencies),
        }
        warms = [r.warm_ms for r in scored if r.warm_ms is not None]
        if warms:
            metrics["warm_ms"] = {"median": statistics.median(warms), "p95": quantile(warms, 0.95)}
        cached = [r.cached_tokens for r in scored if r.cached_tokens is not None]
        prompts = [r.prompt_tokens for r in scored if r.prompt_tokens is not None]
        if cached and prompts:
            metrics["cached_token_share"] = sum(cached) / max(1, sum(prompts))
    return {
        **header,
        "raw_file": raw_path.name,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "metrics": metrics,
        "reliability": reliability,
        "wrong": [
            {
                "case_id": r.case_id,
                "label": r.label,
                "selected": r.selected,
                "confidence": r.confidence,
            }
            for r in scored
            if r.selected != r.label
        ],
        "failed": [{"case_id": r.case_id, "error": r.error} for r in rows if r.error],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    m = summary["metrics"]
    lines = [
        f"# Scorecard: {summary['recipe_stem']} on {summary['suite']}",
        "",
        "Generated by `jevify eval`; every number below derives from the raw file named here.",
        "",
        "| field | value |",
        "|---|---|",
        f"| recipe | `{summary['recipe']}` |",
        f"| recipe_hash | `{summary['recipe_hash']}` |",
        f"| suite | `{summary['suite']}` |",
        f"| suite_sha256 | `{summary['suite_sha256']}` |",
        f"| expected answers by | {summary.get('expected_answers_by', 'unknown')} |",
        f"| model | `{summary['model']}` |",
        f"| backend | {summary['backend'].get('dialect')} "
        f"{summary['backend'].get('version') or ''} |",
        f"| jevify | {summary['jevify_version']} on {summary['python']} |",
        f"| run at | {summary['run_at']} |",
        f"| raw file | `runs/{summary['raw_file']}` (sha256 `{summary['raw_sha256'][:16]}…`) |",
        f"| command | `{summary['command']}` |",
        "",
        "| metric | value |",
        "|---|---|",
        f"| cases | {m['cases']} (scored {m['scored']}, errors {m['errors']}) |",
    ]
    for key in ("accuracy", "log_loss", "brier", "ece", "score_mae", "score_rps", "degraded"):
        if key in m:
            value = m[key]
            lines.append(
                f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |"
            )
    if "latency_ms" in m:
        lat = m["latency_ms"]
        lines.append(f"| latency ms (median / p95) | {lat['median']:.1f} / {lat['p95']:.1f} |")
    if "warm_ms" in m:
        lines.append(
            f"| warm ms (median / p95) | {m['warm_ms']['median']:.1f} / {m['warm_ms']['p95']:.1f} |"
        )
    if "cached_token_share" in m:
        lines.append(f"| cached token share | {m['cached_token_share']:.3f} |")
    if m.get("groups"):
        lines += ["", "| group | cases | accuracy |", "|---|---|---|"]
        for name, g in m["groups"].items():
            lines.append(f"| {name} | {g['cases']} | {g['accuracy']:.4f} |")
    if summary.get("reliability"):
        lines += [
            "",
            "| confidence bin | count | mean confidence | accuracy |",
            "|---|---|---|---|",
        ]
        for cell in summary["reliability"]:
            if cell["count"]:
                lines.append(
                    f"| {cell['lower']:.1f}–{cell['upper']:.1f} | {cell['count']} | "
                    f"{cell['confidence']:.3f} | {cell['accuracy']:.3f} |"
                )
    if summary.get("wrong"):
        lines += ["", "| wrong case | expected | selected | confidence |", "|---|---|---|---|"]
        for w in summary["wrong"]:
            lines.append(
                f"| {w['case_id']} | {w['label']} | {w['selected']} | {w['confidence']:.3f} |"
            )
    if summary.get("failed"):
        lines += ["", "| failed case | error |", "|---|---|"]
        for f in summary["failed"]:
            lines.append(f"| {f['case_id']} | {f['error']} |")
    lines.append("")
    lines.append(
        "Semantics: the probabilities are `readout` distributions unless the recipe carries a "
        "calibration table; accuracy is agreement with the suite's expected answers "
        f"({summary.get('expected_answers_by', 'unknown')}), not truth."
    )
    return "\n".join(lines) + "\n"


def evaluate_to_files(
    engine: Engine,
    recipe: Recipe,
    recipe_path: Path,
    suite: Suite,
    *,
    runs_dir: Path,
    evidence_dir: Path,
    concurrency: int,
    command: str,
) -> dict[str, Any]:
    probe = recipe.probe or {}
    header = {
        "recipe": str(recipe_path),
        "recipe_stem": recipe_path.stem,
        "recipe_hash": recipe_hash(recipe),
        "model": recipe.model.name,
        "backend": {
            "kind": recipe.model.kind,
            "dialect": probe.get("dialect")
            or (recipe.endpoint.dialect if recipe.endpoint else None),
            "version": probe.get("backend_version"),
            "base_url": recipe.endpoint.base_url if recipe.endpoint else None,
            "probed_at": probe.get("probed_at"),
        },
        "suite": suite.name,
        "suite_file": str(suite.path),
        "suite_sha256": suite.sha256,
        "expected_answers_by": suite.manifest.get("expected_answers_by"),
        "jevify_version": __version__,
        "python": platform.python_version(),
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "concurrency": concurrency,
        "command": command,
    }
    decisions = asyncio.run(run_suite(engine, suite, concurrency=concurrency))
    raw_path = write_raw(decisions, runs_dir, recipe, suite, header)
    summary = summarize_raw(raw_path)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{recipe_path.stem}--{suite.name}"
    summary_json = evidence_dir / f"{stem}.json"
    summary_md = evidence_dir / f"{stem}.md"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    return {
        "recipe_hash": header["recipe_hash"],
        "suite_sha256": suite.sha256,
        "raw": str(raw_path),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "metrics": summary["metrics"],
    }
