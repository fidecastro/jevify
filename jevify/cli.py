"""The `jevify` command line: parse, compose, call, print.

Thin by design (ADR-0006 D1, argparse). Every subcommand builds its objects
through `jevify.compose` and prints JSON; behaviour lives in the library.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jevify import __version__
from jevify.api.translate import evaluation_to_jev, jev_extension
from jevify.domain.questions import (
    ChoiceQuestion,
    ImagePart,
    NoulQuestion,
    Option,
    Question,
    ScoreQuestion,
    State,
)
from jevify.ports.backend import BackendError
from jevify.recipes import RecipeError, apply_probe, load_recipe, recipe_hash


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
    ask.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="an image file added to the state after the text; repeatable",
    )
    ask.add_argument(
        "--rung",
        choices=("named_token_logprobs", "grammar", "top_k", "equal_bias", "top_k_floor"),
        help="override the recipe's readout rung for this call",
    )
    probe = commands.add_parser(
        "probe", help="measure a backend and write its capabilities into the recipe"
    )
    probe.add_argument("recipe", type=Path)
    probe.add_argument(
        "--dry-run", action="store_true", help="print the capabilities, do not write"
    )
    return parser


def run_probe(args: argparse.Namespace) -> int:
    from jevify.compose import build_backend
    from jevify.recipes import dump_recipe

    recipe = load_recipe(args.recipe)
    before = recipe_hash(recipe)
    capabilities = asyncio.run(build_backend(recipe).probe())
    updated = apply_probe(recipe, capabilities)
    after = recipe_hash(updated)
    report = {
        "recipe": str(args.recipe),
        "hash_before": before,
        "hash_after": after,
        "capabilities": capabilities.to_dict(),
    }
    if not args.dry_run:
        dump_recipe(updated, args.recipe)
        report["written"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


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


_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def parse_state(args: argparse.Namespace) -> State:
    if args.state_json is not None:
        state = State.from_jev(json.loads(args.state_json))
    else:
        text = args.state
        if text.startswith("@"):
            text = Path(text[1:]).read_text(encoding="utf-8")
        state = State.from_jev(text)
    images = []
    for path in getattr(args, "image", []) or []:
        mime = _IMAGE_TYPES.get(path.suffix.lower())
        if mime is None:
            raise UsageError(f"{path}: unsupported image type; use png, jpg or webp")
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        images.append(ImagePart(f"data:{mime};base64,{payload}"))
    return State(state.parts + tuple(images)) if images else state


def run_ask(args: argparse.Namespace) -> int:
    from jevify.compose import build_engine

    recipe = load_recipe(args.recipe)
    if args.rung:
        recipe = recipe.model_copy(
            update={"readout": recipe.readout.model_copy(update={"rung": args.rung})}
        )
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


COMMANDS = {"ask": run_ask, "probe": run_probe}


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
