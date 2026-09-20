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
    # raw mode only: the placeholder the server substitutes with each image, in order
    image_marker: str = "<__media__>"
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


class RerankSpec(_Strict):
    """How a rerank-kind recipe forms its query and documents."""

    path: str = "rerank"  # under /v1
    query: str = "{state}"
    # placeholders: {key}, {description} (with description_sep), {text} = description or key
    document: str = "{text}"
    description_sep: str = ": "
    instruction: str | None = None  # some rerank APIs take one; sent as `instruction` if set
    score_field: str = "relevance_score"


class LocalModelSpec(_Strict):
    """An in-process model (the `encoder` extra): a Hugging Face id or a local path."""

    path: str
    device: str = "auto"
    max_tokens: int = Field(default=1024, ge=16)
    pooling: Literal["last", "mean"] = "last"


class EmbeddingSpec(_Strict):
    """How an embedding-kind recipe forms the query and the option texts."""

    path: str = "embeddings"  # under /v1, endpoint transport only
    query: str  # the recipe owns the prompt: {instructions}, {state}
    option: str = "{text}"  # {key}, {description}, {text} = description or key
    description_sep: str = ": "
    scale: float = Field(default=10.0, gt=0)  # cosine times scale feeds the ranking softmax


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
    template: TemplateSpec | None = None
    answers: AnswersSpec | None = None
    rerank: RerankSpec | None = None
    embedding: EmbeddingSpec | None = None
    local: LocalModelSpec | None = None
    readout: ReadoutSpec = Field(default_factory=ReadoutSpec)
    budgets: Budgets | None = None
    calibration: Calibration | None = None
    probe: dict[str, Any] | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def _check_kind(self) -> Recipe:
        if self.model.kind in ("endpoint", "rerank") and self.endpoint is None:
            raise ValueError(f"a {self.model.kind} recipe needs an endpoint section")
        if self.model.kind == "embedding":
            if self.embedding is None:
                raise ValueError("an embedding recipe needs an embedding section")
            if (self.endpoint is None) == (self.local is None):
                raise ValueError("an embedding recipe needs exactly one of endpoint or local")
        if self.model.kind == "endpoint" and (self.template is None or self.answers is None):
            raise ValueError("an endpoint recipe needs template and answers sections")
        if self.model.kind == "rerank" and self.rerank is None:
            raise ValueError("a rerank recipe needs a rerank section")
        return self
