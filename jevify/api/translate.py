"""Domain answers to Jev's wire shape, and Jev requests to domain questions.

The one place the shapes `choice`, `score` and `noul` are produced (ADR-0003
D1). Everything jevify adds goes under `x_jevify` (D3). This module imports
no ports and no adapters (a guard enforces it).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jevify.api.schema import (
    ChoiceQuestion as WireChoice,
)
from jevify.api.schema import (
    NoulQuestion as WireNoul,
)
from jevify.api.schema import (
    ScoreQuestion as WireScore,
)
from jevify.api.schema import (
    SystemOneRequest,
)
from jevify.domain.engine import Answer, Evaluation
from jevify.domain.questions import (
    ChoiceQuestion,
    NoulQuestion,
    Option,
    Question,
    ScoreQuestion,
    State,
)


def _text(value: Any) -> str | None:
    """Jev allows instructions and criteria to be strings or JSON; render JSON deterministically."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def jev_request_to_domain(request: SystemOneRequest) -> tuple[State, list[Question]]:
    state = State.from_jev(request.state)
    questions: list[Question] = []
    for ident, wrapped in request.questions.items():
        wire = wrapped.root
        if isinstance(wire, WireNoul):
            criteria = wire.criteria
            questions.append(
                NoulQuestion(
                    id=ident,
                    instructions=_text(wire.instructions),
                    true_criterion=_text(criteria.true) if criteria else None,
                    false_criterion=_text(criteria.false) if criteria else None,
                )
            )
        elif isinstance(wire, WireChoice):
            questions.append(
                ChoiceQuestion(
                    id=ident,
                    instructions=_text(wire.instructions),
                    options=tuple(
                        Option(key, _text(description))
                        for key, description in wire.criteria.items()
                    ),
                )
            )
        elif isinstance(wire, WireScore):
            questions.append(
                ScoreQuestion(
                    id=ident,
                    instructions=_text(wire.instructions),
                    levels=tuple(_text(level) or "" for level in wire.criteria),
                )
            )
    return state, questions


def answer_to_jev(answer: Answer, *, recipe_hash: str) -> dict[str, Any]:
    probabilities = {label: float(p) for label, p in answer.distribution.as_mapping().items()}
    extension: dict[str, Any] = {
        "semantics": answer.semantics,
        "rung": answer.readout.rung,
        "degraded": answer.readout.degraded,
        "off_menu_mass": answer.readout.off_menu_mass,
        "permutation": answer.readout.permutation,
        "calls": answer.readout.calls,
        "latency_ms": answer.readout.latency_ms,
        "prompt_tokens": answer.readout.prompt_tokens,
        "cached_tokens": answer.readout.cached_tokens,
        "recipe_hash": recipe_hash,
    }
    if answer.readout.missing:
        extension["missing"] = list(answer.readout.missing)
    if answer.kind == "choice":
        return {
            "type": "choice",
            "choice": answer.selected,
            "confidence": float(answer.confidence),
            "probabilities": probabilities,
            "x_jevify": extension,
        }
    if answer.kind == "score":
        return {
            "type": "score",
            "score": float(answer.expectation),
            "confidence": float(answer.confidence),
            "legend": dict(answer.legend),
            "probabilities": probabilities,
            "x_jevify": extension,
        }
    extension["confidence"] = float(answer.confidence)
    return {"type": "noul", "noul": float(answer.probability_true), "x_jevify": extension}


def evaluation_to_jev(evaluation: Evaluation, *, model: str, recipe_hash: str) -> dict[str, Any]:
    return {
        "model": model,
        "answers": {
            answer.question_id: answer_to_jev(answer, recipe_hash=recipe_hash)
            for answer in evaluation.answers
        },
        "usage": {
            "input_tokens": int(evaluation.usage.input_tokens),
            "output_tokens": int(evaluation.usage.output_tokens),
        },
    }


def jev_extension(evaluation: Evaluation, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Top-level `x_jevify` for a response: state id and timings."""
    payload: dict[str, Any] = {
        "state_id": evaluation.state_id,
        "warm_ms": evaluation.warm_ms,
        "elapsed_ms": evaluation.elapsed_ms,
    }
    if extra:
        payload.update(extra)
    return payload


def classify_to_systemone(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The convenience route (ADR-0003 D6): context plus categories to a Jev-shaped request.

    `boolean` asks one noul per category; `choice` asks one choice over all categories;
    `score` asks one score per category on the caller's levels. Categories may be a list of
    names or a mapping of name to description.
    """
    context = payload.get("context")
    if context is None:
        raise ValueError("classify needs a context")
    raw_categories = payload.get("categories") or {}
    if isinstance(raw_categories, list):
        categories: dict[str, str | None] = {str(name): None for name in raw_categories}
    elif isinstance(raw_categories, Mapping):
        categories = {str(k): (None if v is None else str(v)) for k, v in raw_categories.items()}
    else:
        raise ValueError("categories must be a list of names or a mapping of name to description")
    if not categories:
        raise ValueError("classify needs at least one category")
    mode = payload.get("mode", "boolean")
    instructions = payload.get("instructions")
    questions: dict[str, Any] = {}
    if mode == "boolean":
        for name, description in categories.items():
            statement = f"The context belongs to the category {name!r}"
            if description:
                statement += f" ({description})"
            statement += "."
            if instructions:
                statement = f"{instructions} {statement}"
            questions[name] = {"type": "noul", "instructions": statement}
    elif mode == "choice":
        questions["category"] = {
            "type": "choice",
            "instructions": instructions or "Which category does the context belong to?",
            "criteria": dict(categories),
        }
    elif mode == "score":
        levels = payload.get("levels")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError("score mode needs at least two ordered levels")
        for name, description in categories.items():
            statement = f"How strongly does the context match the category {name!r}"
            if description:
                statement += f" ({description})"
            statement += "?"
            if instructions:
                statement = f"{instructions} {statement}"
            questions[name] = {"type": "score", "instructions": statement, "criteria": list(levels)}
    else:
        raise ValueError(f"unknown mode {mode!r}; use boolean, choice or score")
    request: dict[str, Any] = {
        "state": context,
        "model": str(payload.get("model") or "jevify"),
        "questions": questions,
    }
    if payload.get("x_jevify"):
        request["x_jevify"] = payload["x_jevify"]
    return request


def classify_results(
    mode: str, categories: list[str], jev_answers: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-category results derived from the Jev-shaped answers (no second readout)."""
    results: dict[str, Any] = {}
    if mode == "choice":
        answer = jev_answers["category"]
        for name in categories:
            probability = float(answer["probabilities"].get(name, 0.0))
            results[name] = {
                "probability": probability,
                "decision": answer["choice"] == name,
                "x_jevify": answer["x_jevify"],
            }
        return results
    for name in categories:
        answer = jev_answers[name]
        if mode == "boolean":
            probability = float(answer["noul"])
            results[name] = {
                "probability": probability,
                "decision": probability >= 0.5,
                "x_jevify": answer["x_jevify"],
            }
        else:
            results[name] = {
                "score": float(answer["score"]),
                "legend": answer["legend"],
                "probabilities": answer["probabilities"],
                "x_jevify": answer["x_jevify"],
            }
    return results
