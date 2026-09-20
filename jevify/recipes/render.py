"""The prompt chokepoint (ADR-0004 D3, INV-6).

Nothing else in jevify turns a state or a question into text. Every
placeholder is filled by replacement, never by `str.format`, so braces inside
a state or an instruction can never be mistaken for placeholders.

A rendered question is one or more *parts*. With identifier-based choice the
question is a single part whose answer set is the identifiers; with the
per-option strategy each option is its own yes/no part and the adapter
composes them (ADR-0002 D2: state as shared prefix, each option a suffix).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from jevify.domain.questions import (
    ChoiceQuestion,
    ImagePart,
    NoulQuestion,
    Question,
    QuestionKind,
    ScoreQuestion,
    State,
)
from jevify.recipes.schema import Recipe, Token
from jevify.recipes.store import RecipeError

_STATE = "{state}"
_QUESTION = "{question}"

Composition = Literal["single", "per_option"]


@dataclass(frozen=True)
class RenderedPrefix:
    """The state-only part of a prompt: identical across every question about that state."""

    mode: str
    text: str
    system: str | None = None
    kwargs: dict[str, object] = field(default_factory=dict)
    images: tuple[ImagePart, ...] = ()
    prefill: str | None = None


@dataclass(frozen=True)
class RenderedPart:
    """One request's suffix, ending at the answer slot, with its own answer set."""

    text: str
    labels: tuple[str, ...]
    tokens: dict[str, tuple[Token, ...]]
    label: str | None = None  # the option this part scores, under per_option composition


@dataclass(frozen=True)
class RenderedQuestion:
    kind: QuestionKind
    labels: tuple[str, ...]
    composition: Composition
    parts: tuple[RenderedPart, ...]
    order: tuple[int, ...] = ()

    @property
    def text(self) -> str:
        """The suffix of a single-part question."""
        if self.composition != "single":
            raise RecipeError("a per-option question has one suffix per option, not one text")
        return self.parts[0].text

    @property
    def tokens(self) -> dict[str, tuple[Token, ...]]:
        if self.composition != "single":
            raise RecipeError("a per-option question has one answer set per part")
        return self.parts[0].tokens


def _fill(template: str, **fields: str) -> str:
    for name, value in fields.items():
        template = template.replace("{" + name + "}", value)
    return template


def _identifier_text(group: Sequence[Token]) -> str:
    return group[0].text.strip()


def render_prefix(recipe: Recipe, state: State) -> RenderedPrefix:
    template = recipe.template
    images = tuple(part for part in state.parts if isinstance(part, ImagePart))
    if template.mode == "raw":
        assert template.raw is not None  # validated by the schema
        head = template.raw[: template.raw.index(_QUESTION)]
        text = _fill(head, state=state.text) + template.state_end
        return RenderedPrefix(mode="raw", text=text, images=images)
    text = _fill(template.state, state=state.text) + template.state_end
    return RenderedPrefix(
        mode="messages",
        text=text,
        system=template.system,
        kwargs=dict(template.kwargs),
        images=images,
        prefill=template.prefill,
    )


def render_question(
    recipe: Recipe, question: Question, order: tuple[int, ...] | None = None
) -> RenderedQuestion:
    templates = recipe.template.questions
    instructions = question.instructions or ""
    noul_tokens = {
        "false": tuple(recipe.answers.noul["false"]),
        "true": tuple(recipe.answers.noul["true"]),
    }

    if isinstance(question, NoulQuestion):
        body = _fill(
            templates.noul,
            instructions=instructions,
            true_criterion=question.true_criterion or "",
            false_criterion=question.false_criterion or "",
        )
        part = RenderedPart(_tail(recipe, body), ("false", "true"), noul_tokens)
        return RenderedQuestion("noul", ("false", "true"), "single", (part,))

    if isinstance(question, ChoiceQuestion):
        items = [(option.key, option.description) for option in question.options]
        line_template, label_of = templates.option_line, "key"
    elif isinstance(question, ScoreQuestion):
        items = [(level, None) for level in question.levels]
        line_template, label_of = templates.level_line, "text"
    else:  # pragma: no cover - the union is closed
        raise RecipeError(f"unsupported question type {type(question).__name__}")
    labels = question.labels

    if templates.choice_strategy == "per_option":
        assert templates.option_document is not None  # validated by the schema
        parts = []
        for label, (text, description) in zip(labels, items, strict=True):
            fields = {
                "instructions": instructions,
                label_of: text,
                "key": text,
                "description": (templates.description_sep + description) if description else "",
            }
            body = _fill(templates.option_document, **fields)
            parts.append(RenderedPart(_tail(recipe, body), ("false", "true"), noul_tokens, label))
        return RenderedQuestion(question.kind, labels, "per_option", tuple(parts))

    alphabet = recipe.answers.identifiers
    if len(items) > len(alphabet):
        raise RecipeError(
            f"{question.kind} question {question.id!r} has {len(items)} options but the recipe "
            f"verifies {len(alphabet)} identifiers; split the menu or use an encoder recipe"
        )
    display = tuple(order) if order is not None else tuple(range(len(items)))
    if sorted(display) != list(range(len(items))):
        raise RecipeError(f"order must be a permutation of {len(items)} option positions")

    lines: list[str] = []
    tokens: dict[str, tuple[Token, ...]] = {}
    for position, item_index in enumerate(display):
        text, description = items[item_index]
        group = alphabet[position]
        fields = {
            "identifier": _identifier_text(group),
            label_of: text,
            "description": (templates.description_sep + description) if description else "",
        }
        lines.append(_fill(line_template, **fields))
        tokens[labels[item_index]] = tuple(group)
    block = "\n".join(lines)
    template = templates.choice if isinstance(question, ChoiceQuestion) else templates.score
    body = _fill(template, instructions=instructions, options=block, levels=block)
    part = RenderedPart(_tail(recipe, body), labels, tokens)
    return RenderedQuestion(question.kind, labels, "single", (part,), display)


def _tail(recipe: Recipe, body: str) -> str:
    """In raw mode the suffix carries the template's tail up to the answer slot."""
    if recipe.template.mode == "raw":
        assert recipe.template.raw is not None
        tail = recipe.template.raw[recipe.template.raw.index(_QUESTION) :]
        return _fill(tail, question=body)
    return body
