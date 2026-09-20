"""The `jevify` command line: parse, compose, call, print.

Thin by design (ADR-0006 D1, argparse). Every subcommand builds its objects
through `jevify.compose` and prints JSON; behaviour lives in the library.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from jevify import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevify",
        description=(
            "Make a pretrained model you already have answer typed questions in one pass, "
            "and measure how well it does."
        ),
    )
    parser.add_argument("--version", action="version", version=f"jevify {__version__}")
    parser.add_subparsers(dest="command", metavar="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return 0  # subcommands are added slice by slice


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    sys.exit(main())
