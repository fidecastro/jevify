"""Suites: frozen, hashed JSONL cases with a manifest (ADR-0005 D1).

Row format, carried from jevlike and decision-poc: one JSON object per line
with `instruction`, `context`, `options`, optional `label` (index into
options), `case_id`, optional `group` and optional `question_type`
(choice by default; noul rows use options ["false", "true"]; score rows
list ordered levels as options).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class SuiteError(ValueError):
    """A suite file or manifest is invalid; the message names the line or field."""


QuestionType = Literal["choice", "noul", "score"]


@dataclass(frozen=True)
class SuiteRow:
    case_id: str
    instruction: str
    context: str
    options: tuple[str, ...]
    label: int | None
    group: str = "default"
    question_type: QuestionType = "choice"


@dataclass(frozen=True)
class Suite:
    path: Path
    rows: tuple[SuiteRow, ...]
    manifest: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or self.path.stem)

    @property
    def sha256(self) -> str:
        return str(self.manifest["sha256"])


def read_rows(path: str | Path) -> list[SuiteRow]:
    rows: list[SuiteRow] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SuiteError(f"line {number}: not valid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise SuiteError(f"line {number}: expected an object")
            for field in ("instruction", "context"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise SuiteError(f"line {number}: nonempty {field} required")
            options = row.get("options")
            if (
                not isinstance(options, list)
                or len(options) < 2
                or any(not isinstance(o, str) or not o.strip() for o in options)
                or len(set(options)) != len(options)
            ):
                raise SuiteError(f"line {number}: need at least two distinct text options")
            label = row.get("label")
            if label is not None and (type(label) is not int or not 0 <= label < len(options)):
                raise SuiteError(f"line {number}: invalid label")
            case_id = str(row.get("case_id") or f"row_{number}")
            if case_id in seen:
                raise SuiteError(f"line {number}: duplicate case_id {case_id!r}")
            seen.add(case_id)
            question_type = row.get("question_type", "choice")
            if question_type not in ("choice", "noul", "score"):
                raise SuiteError(f"line {number}: unknown question_type {question_type!r}")
            rows.append(
                SuiteRow(
                    case_id=case_id,
                    instruction=row["instruction"],
                    context=row["context"],
                    options=tuple(options),
                    label=label,
                    group=str(row.get("group") or "default"),
                    question_type=question_type,
                )
            )
    if not rows:
        raise SuiteError(f"{path}: empty suite")
    return rows


def suite_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(source.stem + ".manifest.json")


def load_suite(path: str | Path) -> Suite:
    """Load rows and manifest; refuse a file whose hash no longer matches its manifest."""
    source = Path(path)
    manifest_file = manifest_path(source)
    if not manifest_file.exists():
        raise SuiteError(f"{source}: no manifest at {manifest_file}; a suite is frozen and hashed")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    actual = suite_hash(source)
    if manifest.get("sha256") != actual:
        raise SuiteError(
            f"{source}: sha256 {actual[:12]} does not match the manifest's "
            f"{str(manifest.get('sha256'))[:12]}; a changed suite is a new suite"
        )
    rows = read_rows(source)
    if manifest.get("cases") not in (None, len(rows)):
        raise SuiteError(f"{source}: manifest says {manifest['cases']} cases, file has {len(rows)}")
    return Suite(path=source, rows=tuple(rows), manifest=manifest)
