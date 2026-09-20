"""The command-line seam: `jevify` as an installed console script."""

from __future__ import annotations

import subprocess
import sys


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
