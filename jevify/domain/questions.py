"""Value types for state and typed questions (docs/00-invariants.md §2)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

Semantics = Literal["readout", "relevance", "similarity", "calibrated"]
QuestionKind = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """An image as a data URI. jevify never fetches URLs (ADR-0003 D5)."""

    data_uri: str


Part = TextPart | ImagePart


@dataclass(frozen=True)
class State:
    """The shared input every question is asked about."""

    parts: tuple[Part, ...]

    @classmethod
    def from_jev(cls, value: str | Mapping[str, object] | Sequence[object]) -> State:
        """Build from Jev's `state` field: a string, a JSON object or array, or a list of
        content parts in the chat-endpoint shape (`text` and `image_url` parts).

        Structured values render deterministically as indented JSON with key order
        preserved, so the same state always produces the same prefix. Images must be
        data URIs: jevify never fetches a URL on a caller's behalf (ADR-0003 D5).
        """
        if isinstance(value, str):
            return cls((TextPart(value),))
        if isinstance(value, Sequence) and value and all(_is_content_part(v) for v in value):
            parts: list[Part] = []
            for item in value:
                assert isinstance(item, Mapping)
                if item["type"] == "text":
                    parts.append(TextPart(str(item.get("text", ""))))
                else:
                    image = item.get("image_url")
                    url = image.get("url") if isinstance(image, Mapping) else image
                    if not isinstance(url, str) or not url.startswith("data:"):
                        raise ValueError(
                            "image parts must be data URIs; jevify never fetches a remote URL"
                        )
                    parts.append(ImagePart(url))
            return cls(tuple(parts))
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
        return cls((TextPart(rendered),))

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))

    @property
    def modalities(self) -> frozenset[str]:
        kinds = {"text" if isinstance(p, TextPart) else "image" for p in self.parts}
        return frozenset(kinds)


def _is_content_part(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("type") in ("text", "image_url")


@dataclass(frozen=True)
class Option:
    key: str
    description: str | None = None


@dataclass(frozen=True)
class ChoiceQuestion:
    id: str
    instructions: str | None
    options: tuple[Option, ...]
    kind: QuestionKind = field(default="choice", init=False)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(option.key for option in self.options)


@dataclass(frozen=True)
class ScoreQuestion:
    id: str
    instructions: str | None
    levels: tuple[str, ...]
    kind: QuestionKind = field(default="score", init=False)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(str(index) for index in range(len(self.levels)))


@dataclass(frozen=True)
class NoulQuestion:
    id: str
    instructions: str | None
    true_criterion: str | None = None
    false_criterion: str | None = None
    kind: QuestionKind = field(default="noul", init=False)

    @property
    def labels(self) -> tuple[str, ...]:
        return ("false", "true")


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion
