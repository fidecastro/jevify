"""The command-line seam: `jevify` as an installed console script."""

from __future__ import annotations

import json
import math
import subprocess
import sys

from tests.fakes.fake_openai_server import Behaviour


def test_ask_prints_jev_shaped_json(fixtures, fake_http_server_factory, tmp_path, capsys) -> None:
    """`jevify ask <recipe> --state ... --noul ... --choice ...` prints Jev-shaped answers."""
    import yaml

    from jevify.cli import main

    live = fake_http_server_factory(
        Behaviour(scores={" yes": 0.0, " no": -2.0, " A": -1.0, " B": 0.0})
    )
    doc = yaml.safe_load((fixtures / "recipes" / "fake-vllm.yaml").read_text())
    doc["endpoint"]["base_url"] = live.base_url
    recipe_path = tmp_path / "live-fake.yaml"
    recipe_path.write_text(yaml.safe_dump(doc))

    code = main(
        [
            "ask",
            str(recipe_path),
            "--state",
            "The customer wrote: keep my subscription, stop the newsletters.",
            "--noul",
            "cancel:The customer wants to cancel.",
            "--choice",
            "dept:Which team handles this?=billing,support",
        ]
    )
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["model"] == "fake-decoder"
    assert body["answers"]["cancel"]["type"] == "noul"
    assert round(body["answers"]["cancel"]["noul"], 3) == round(1 / (1 + math.exp(-2)), 3)
    assert body["answers"]["dept"]["choice"] == "support"
    assert body["answers"]["dept"]["x_jevify"]["semantics"] == "readout"
    assert body["usage"]["output_tokens"] == 2
    assert body["x_jevify"]["recipe_hash"]
    assert len(live.server.requests) == 2


def test_version_prints_package_version() -> None:
    """`jevify --version` prints the package version on stdout and exits 0.

    Runs the module through the interpreter so the test exercises the same
    entry point the console script does, without depending on PATH.
    """
    import jevify

    result = subprocess.run(
        [sys.executable, "-m", "jevify", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"jevify {jevify.__version__}"
