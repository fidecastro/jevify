"""Live proof: `jevify ask` against every recipe named in JEVIFY_LIVE_RECIPES.

Skipped unless the variable is set (a comma-separated list of recipe paths).
These tests assert shape and sanity, not accuracy; accuracy is a scorecard's job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from jevify.cli import main

RECIPES = [Path(p) for p in os.environ.get("JEVIFY_LIVE_RECIPES", "").split(",") if p.strip()]

pytestmark = pytest.mark.live


@pytest.mark.skipif(not RECIPES, reason="JEVIFY_LIVE_RECIPES not set")
@pytest.mark.parametrize("recipe", RECIPES, ids=[p.stem for p in RECIPES])
def test_live_ask_returns_readout_distributions(recipe: Path, capsys) -> None:
    code = main(
        [
            "ask",
            str(recipe),
            "--state",
            "The customer wrote: I was charged twice this month and I am furious. "
            "If this is not fixed today I will cancel.",
            "--noul",
            "threatens_cancel:The customer threatens to cancel.",
            "--noul",
            "pleased:The customer is pleased with the service.",
            "--choice",
            "dept:Which team should handle this?=billing,technical support,sales",
        ]
    )
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    answers = body["answers"]
    assert answers["threatens_cancel"]["noul"] > 0.5
    assert answers["pleased"]["noul"] < 0.5
    assert answers["dept"]["choice"] == "billing"
    for answer in answers.values():
        assert answer["x_jevify"]["semantics"] == "readout"
        assert answer["x_jevify"]["degraded"] is False
    assert body["usage"]["output_tokens"] == 3
