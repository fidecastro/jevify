"""Jev's wire schema (ADR-0003 D1), mirrored from typesafe-sdk 0.7.0 and pinned by a guard.

Field names, nesting and required-ness are Jev's. Everything jevify adds is
optional and lives under `x_jevify` (D3). Response models are strict so that
an int where the SDK expects a float turns a test red here, not on a client.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

JSONContent = str | dict[str, Any] | list[Any]


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ requests
class NoulCriteria(_Wire):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(_Wire):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Wire):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]


class ScoreQuestion(_Wire):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(min_length=1)


class Question(RootModel[NoulQuestion | ChoiceQuestion | ScoreQuestion]):
    root: NoulQuestion | ChoiceQuestion | ScoreQuestion = Field(discriminator="type")


class SystemOneRequest(BaseModel):
    """Jev's request plus the optional `x_jevify` extension (state_ref, images...)."""

    model_config = ConfigDict(extra="forbid")

    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    x_jevify: dict[str, Any] | None = None


# ------------------------------------------------------------------ responses
class _StrictWire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoulAnswer(_StrictWire):
    type: Literal["noul"]
    noul: float
    x_jevify: dict[str, Any] | None = None


class ChoiceAnswer(_StrictWire):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]
    x_jevify: dict[str, Any] | None = None


class ScoreAnswer(_StrictWire):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]
    x_jevify: dict[str, Any] | None = None


class Answer(RootModel[NoulAnswer | ChoiceAnswer | ScoreAnswer]):
    root: NoulAnswer | ChoiceAnswer | ScoreAnswer = Field(discriminator="type")


class Usage(_StrictWire):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(_StrictWire):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    x_jevify: dict[str, Any] | None = None


class ModelMetadata(_StrictWire):
    name: str
    description: str
    release_date: str


class ModelMetadataList(_StrictWire):
    models: list[ModelMetadata]


class ValidationError(_Wire):
    loc: list[str | int]
    msg: str
    type: str
    input: Any | None = None
    ctx: dict[str, Any] | None = None


class HTTPValidationError(_Wire):
    detail: list[ValidationError] | None = None
