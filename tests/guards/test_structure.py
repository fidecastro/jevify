"""Structure guards: the ratchet tests that keep chokepoints structural (AGENTS.md §D).

Each guard scans the tree with `ast` and, when it fails, names the chokepoint
to use. Guards are anti-accident, not anti-adversary.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "jevify"

# Packages that form the core: they depend on ports, never on adapters or on
# any model or transport library (ADR-0002 D1, AGENTS.md §C "D").
CORE_PACKAGES = ("domain", "ports", "recipes", "evaluation", "api")
FORBIDDEN_IN_CORE = ("jevify.adapters", "httpx", "torch", "transformers", "uvicorn")
# api/app.py is the one core module allowed to import the web framework.
STARLETTE_ALLOWED = {PACKAGE / "api" / "app.py"}


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_guard_core_imports_no_adapters() -> None:
    """Core packages import ports, never adapters, httpx, torch, transformers or uvicorn.

    Chokepoint: dependencies on infrastructure are wired in `jevify/compose.py`
    and reach the core through the Protocols in `jevify/ports/`.
    """
    offenders: list[str] = []
    for package in CORE_PACKAGES:
        root = PACKAGE / package
        if not root.exists():
            continue
        for path in _python_files(root):
            for name in _imports(path):
                if any(name == f or name.startswith(f + ".") for f in FORBIDDEN_IN_CORE):
                    offenders.append(f"{path.relative_to(PACKAGE.parent)} imports {name}")
                if name.startswith("starlette") and path not in STARLETTE_ALLOWED:
                    offenders.append(f"{path.relative_to(PACKAGE.parent)} imports {name}")
    assert not offenders, (
        "Core code must depend on jevify.ports, not on adapters or infrastructure. "
        "Wire infrastructure in jevify/compose.py. Offenders:\n  " + "\n  ".join(offenders)
    )
