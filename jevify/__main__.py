"""`python -m jevify` runs the same entry point as the `jevify` console script."""

from __future__ import annotations

import sys

from jevify.cli import main

sys.exit(main())
