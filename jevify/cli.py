"""The `jevify` command line: parse, compose, call, print.

Thin by design (ADR-0006 D1, argparse). Every subcommand builds its objects
through `jevify.compose` and prints JSON; behaviour lives in the library.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jevify import __version__
from jevify.api.translate import evaluation_to_jev, jev_extension
from jevify.domain.questions import (
    ChoiceQuestion,
    NoulQuestion,
    Option,
    Question,
    ScoreQuestion,
    State,
)
from jevify.ports.backend import BackendError
from jevify.recipes import RecipeError, load_recipe, recipe_hash


class UsageError(ValueError):
    """The command line was malformed; the message says how to write it."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevify",
        description=(
            "Make a pretrained model you already have answer typed questions in one pass, "
            "and measure how well it does."
        ),
    )
    parser.add_argument("--version", action="version", version=f"jevify {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="command")

    ask = commands.add_parser("ask", help="answer typed questions about one state")
    ask.add_argument("recipe", type=Path, help="recipe file (how the model is asked)")
    state = ask.add_mutually_exclusive_group(required=True)
    state.add_argument("--state", help="state text, or @path to read it from a file")
    state.add_argument("--state-json", help="state as a JSON object or array")
    ask.add_argument(
        "--choice",
        action="append",
        default=[],
        metavar="ID:INSTRUCTIONS=OPT1,OPT2,...",
        help="a choice question; repeatable",
    )
    ask.add_argument(
        "--score",
        action="append",
        default=[],
        metavar="ID:INSTRUCTIONS=LEVEL0,LEVEL1,...",
        help="a score question with ordered levels; repeatable",
    )
    ask.add_argument(
        "--noul",
        action="append",
        default=[],
        metavar="ID:STATEMENT",
        help="a noul question: probability that the statement holds; repeatable",
    )
    return parser


def parse_questions(args: argparse.Namespace) -> list[Question]:
    questions: list[Question] = []
    for spec in args.choice:
        ident, instructions, items = _split_spec(spec, want_items=True)
        questions.append(
            ChoiceQuestion(
                id=ident, instructions=instructions, options=tuple(Option(key) for key in items)
            )
        )
    for spec in args.score:
        ident, instructions, items = _split_spec(spec, want_items=True)
        questions.append(ScoreQuestion(id=ident, instructions=instructions, levels=tuple(items)))
    for spec in args.noul:
        ident, instructions, _ = _split_spec(spec, want_items=False)
        questions.append(NoulQuestion(id=ident, instructions=instructions))
    if not questions:
        raise UsageError("give at least one --choice, --score or --noul question")
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise UsageError(f"question id {question.id!r} is used twice")
        seen.add(question.id)
    return questions


def _split_spec(spec: str, *, want_items: bool) -> tuple[str, str, list[str]]:
    if ":" not in spec:
        raise UsageError(f"{spec!r}: expected ID:INSTRUCTIONS" + ("=A,B,..." if want_items else ""))
    ident, rest = spec.split(":", 1)
    if not ident.strip():
        raise UsageError(f"{spec!r}: the question id is empty")
    if not want_items:
        return ident.strip(), rest.strip(), []
    if "=" not in rest:
        raise UsageError(f"{spec!r}: expected ID:INSTRUCTIONS=A,B,...")
    instructions, items = rest.rsplit("=", 1)
    parsed = [item.strip() for item in items.split(",") if item.strip()]
    if len(parsed) < 2:
        raise UsageError(f"{spec!r}: give at least two comma-separated options")
    return ident.strip(), instructions.strip(), parsed


def parse_state(args: argparse.Namespace) -> State:
    if args.state_json is not None:
        return State.from_jev(json.loads(args.state_json))
    text = args.state
    if text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8")
    return State.from_jev(text)


def run_ask(args: argparse.Namespace) -> int:
    from jevify.compose import build_engine

    recipe = load_recipe(args.recipe)
    questions = parse_questions(args)
    state = parse_state(args)
    engine = build_engine(recipe)
    evaluation = asyncio.run(engine.ask(state, questions))
    digest = recipe_hash(recipe)
    body = evaluation_to_jev(evaluation, model=recipe.model.name, recipe_hash=digest)
    body["x_jevify"] = jev_extension(
        evaluation,
        {
            "recipe": str(args.recipe),
            "recipe_hash": digest,
            "base_url": recipe.endpoint.base_url if recipe.endpoint else None,
        },
    )
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return 0


COMMANDS = {"ask": run_ask}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return COMMANDS[args.command](args)
    except (UsageError, RecipeError, BackendError) as exc:
        print(f"jevify: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    sys.exit(main())
