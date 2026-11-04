"""Guard: the rung names have one registry, `jevify.ports.backend.RUNG_RANK`.

The recipe schema cannot import the port (the port imports the recipe renderer, which
imports the schema), so its `Rung` literal is a pinned mirror; the CLI's `--rung` choices
are derived. A rung added to the port but not to the mirror, or the reverse, turns this red.
"""

from __future__ import annotations

from typing import get_args

from jevify.cli import build_parser
from jevify.ports.backend import RUNG_RANK
from jevify.recipes.schema import Rung as RecipeRung


def test_rung_names_are_derived_from_the_one_registry():
    """The recipe schema's rung literal and the CLI's --rung choices mirror RUNG_RANK; a rung
    added to the port but not to them (or the reverse) turns this red."""
    registry = tuple(str(r) for r in RUNG_RANK)
    assert get_args(RecipeRung) == ("auto", *registry)
    ask = next(
        action
        for action in build_parser()._subparsers._group_actions[0].choices["ask"]._actions
        if "--rung" in action.option_strings
    )
    assert tuple(ask.choices) == registry
