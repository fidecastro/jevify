"""Guard: a model is asked only through a recipe (INV-6, ADR-0004 D3).

No string literal in `jevify/` may look like a prompt fragment. The only
places prompt text may live are recipe files (`recipes/`, the probe case,
test fixtures). False positives are resolved by moving the string into a
recipe, never by widening this allowlist.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "jevify"

PROMPT_PATTERNS = [
    re.compile(p)
    for p in (
        r"<\|",  # chat control tokens such as <|im_start|>, <｜User｜>
        r"<think>",
        r"</think>",
        r"^Answer( with|:)",
        r"^Question:",
        r"^You are ",
        r"^Instruct:",
        r"<Instruct>",
        r"<Query>",
        r"<Document>",
    )
]


def _string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_guard_no_prompt_literals_outside_recipes() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for literal in _string_literals(path):
            for line in literal.splitlines():
                if any(p.search(line) for p in PROMPT_PATTERNS):
                    offenders.append(f"{path.relative_to(PACKAGE.parent)}: {line[:60]!r}")
    assert not offenders, (
        "Prompt text belongs in a recipe file (recipes/, jevify/recipes/probe_case.yaml, "
        "tests/fixtures/), never in code. Offenders:\n  " + "\n  ".join(offenders)
    )
