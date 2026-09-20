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


def test_guard_classify_translation_never_touches_the_port() -> None:
    """ADR-0003 D6: the classify route is a translation onto the Jev-shaped call, never a
    second implementation; jevify/api/translate.py may import domain and schema only."""
    path = PACKAGE / "api" / "translate.py"
    offenders = [
        name
        for name in _imports(path)
        if name.startswith("jevify.ports") or name.startswith("jevify.adapters")
    ]
    assert not offenders, (
        "jevify/api/translate.py must translate to and from the domain only; "
        f"it imports {offenders}"
    )


def test_guard_every_committed_recipe_loads() -> None:
    """A recipe in recipes/ is a public artifact; a broken one is a broken build."""
    from jevify.recipes import load_recipe

    folder = PACKAGE.parent / "recipes"
    paths = sorted(folder.glob("*.yaml"))
    assert paths, "recipes/ must hold at least one recipe"
    for path in paths:
        recipe = load_recipe(path)
        assert recipe.provenance.status in ("draft", "probed", "scored"), path
