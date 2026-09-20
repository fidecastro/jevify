"""The recipe schema (ADR-0004 D2). Validation errors name the field."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Rung = Literal["auto", "named_token_logprobs", "grammar", "top_k", "equal_bias", "top_k_floor"]
Dialect = Literal["auto", "vllm", "llamacpp", "generic"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Token(_Strict):
    """One spelling of an answer at the answer slot, with its id once the probe verified it."""

    text: str
    id: int | None = None


class ModelSpec(_Strict):
    name: str
    kind: Literal["endpoint", "encoder", "rerank", "embedding"]
    revision: str | None = None


class EndpointSpec(_Strict):
    base_url: str
    api_key_env: str | None = None
    dialect: Dialect = "auto"
    timeout_s: float = Field(default=120.0, gt=0)


class QuestionTemplates(_Strict):
    """How each question type is written after the state. Placeholders in braces."""

    choice: str
    option_line: str
    description_sep: str = ": "
    score: str
    level_line: str
    noul: str
    # `identifiers`: one request, options labelled with single-token identifiers.
    # `per_option`: one yes/no request per option, the option rendered as `option_document`;
    # the model class that never emits identifiers (a yes/no reranker) needs this.
    choice_strategy: Literal["identifiers", "per_option"] = "identifiers"
    option_document: str | None = None

    @model_validator(mode="after")
    def _check_strategy(self) -> QuestionTemplates:
        if self.choice_strategy == "per_option" and not self.option_document:
            raise ValueError("choice_strategy per_option needs option_document")
        return self


class TemplateSpec(_Strict):
    mode: Literal["messages", "raw"]
    system: str | None = None
    state: str = "{state}"
    state_end: str = ""
    raw: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    prefill: str | None = None
    questions: QuestionTemplates

    @model_validator(mode="after")
    def _check_mode(self) -> TemplateSpec:
        if self.mode == "raw":
            if not self.raw:
                raise ValueError("raw mode needs template.raw")
            state_at, question_at = self.raw.find("{state}"), self.raw.find("{question}")
            if state_at < 0 or question_at < 0:
                raise ValueError("template.raw must contain {state} and {question}")
            if question_at < state_at:
                raise ValueError("template.raw must place {state} before {question}")
        elif "{state}" not in self.state:
            raise ValueError("template.state must contain {state}")
        return self


class AnswersSpec(_Strict):
    noul: dict[Literal["true", "false"], list[Token]]
    identifiers: list[list[Token]] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_noul(self) -> AnswersSpec:
        for label in ("true", "false"):
            if not self.noul.get(label):
                raise ValueError(f"answers.noul.{label} needs at least one token")
        if any(not group for group in self.identifiers):
            raise ValueError("every identifier needs at least one token")
        return self


class ReadoutSpec(_Strict):
    rung: Rung = "auto"
    top_k: int = Field(default=20, ge=1)
    permutation: Literal["none", "cyclic", "full"] = "none"
    permutation_calls: int = Field(default=1, ge=1)


class Budgets(_Strict):
    max_context_tokens: int | None = None
    max_options: int | None = None


class Calibration(_Strict):
    temperatures: dict[str, float]
    evidence: str | None = None


class Provenance(_Strict):
    author: str
    created: str
    status: Literal["draft", "probed", "scored"] = "draft"
    launch: str | None = None
    notes: list[str] = Field(default_factory=list)
    scorecards: list[str] = Field(default_factory=list)


class Recipe(_Strict):
    schema_version: Literal[1]
    model: ModelSpec
    endpoint: EndpointSpec | None = None
    template: TemplateSpec
    answers: AnswersSpec
    readout: ReadoutSpec = Field(default_factory=ReadoutSpec)
    budgets: Budgets | None = None
    calibration: Calibration | None = None
    probe: dict[str, Any] | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def _check_endpoint(self) -> Recipe:
        if self.model.kind == "endpoint" and self.endpoint is None:
            raise ValueError("an endpoint recipe needs an endpoint section")
        return self
