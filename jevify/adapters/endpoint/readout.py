"""The readout ladder (ADR-0002 D4): how a distribution over the answer set is obtained.

Each rung knows whether it is available under a capability list, how to
shape the request, and how to read the answer tokens out of the response.
The token match is id-first, text second (ADR-0002 D2, plan decision 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from jevify.adapters.endpoint.dialects import Entry, Parsed, grammar_for, neutral_samplers
from jevify.ports.backend import BackendError, Capabilities, Dialect, Rung
from jevify.recipes.render import RenderedPart
from jevify.recipes.schema import Token

_PIECE_MARKS = {"Ġ": " ", "Ċ": "\n", "▁": " "}


def normalize_piece(text: str) -> str:
    for mark, plain in _PIECE_MARKS.items():
        text = text.replace(mark, plain)
    return text


def matches(entry: Entry, token: Token) -> bool:
    if entry.id is not None and token.id is not None:
        return entry.id == token.id
    return normalize_piece(entry.text) == normalize_piece(token.text)


@dataclass(frozen=True)
class Readout:
    logprobs: dict[str, float]
    off_menu_mass: float | None
    missing: tuple[str, ...]
    degraded: bool


def read_answer_set(parsed: Parsed, question: RenderedPart, *, full_softmax: bool) -> Readout:
    """Collect each label's mass from the entries; a label's tokens are distinct events."""
    per_label: dict[str, float] = {}
    answer_mass = 0.0
    missing: list[str] = []
    for label in question.labels:
        found = [
            entry.logprob
            for entry in parsed.entries
            if any(matches(entry, token) for token in question.tokens[label])
        ]
        if not found:
            missing.append(label)
            continue
        peak = max(found)
        logsum = peak + math.log(sum(math.exp(lp - peak) for lp in found))
        per_label[label] = logsum
        answer_mass += math.exp(logsum)
    off_menu = max(0.0, 1.0 - answer_mass) if full_softmax and not missing else None
    return Readout(per_label, off_menu, tuple(missing), degraded=bool(missing))


class TopKRung:
    """Rung 3: ask for the top-k list unbiased; exact when every answer token is present."""

    rung = Rung.TOP_K

    @staticmethod
    def available(capabilities: Capabilities | None) -> bool:
        return True  # every OpenAI-compatible server lists top logprobs

    @staticmethod
    def shape(body: dict[str, Any], question: RenderedPart, top_k: int, dialect: Dialect) -> None:
        if "messages" in body:
            body["top_logprobs"] = top_k
        else:
            body["logprobs"] = top_k

    @staticmethod
    def read(parsed: Parsed, question: RenderedPart) -> Readout:
        readout = read_answer_set(parsed, question, full_softmax=True)
        if readout.missing:
            raise BackendError(
                f"labels {list(readout.missing)} not in the top-{len(parsed.entries)} logprobs; "
                "raise readout.top_k or pin readout.rung to top_k_floor"
            )
        return readout


class NamedTokenLogprobsRung:
    """Rung 1: ask vLLM for exactly the answer token ids; exact, off-menu mass visible."""

    rung = Rung.NAMED_TOKEN_LOGPROBS

    @staticmethod
    def available(capabilities: Capabilities | None) -> bool:
        return bool(capabilities and Rung.NAMED_TOKEN_LOGPROBS in capabilities.rungs)

    @staticmethod
    def shape(body: dict[str, Any], part: RenderedPart, top_k: int, dialect: Dialect) -> None:
        ids = sorted({t.id for lbl in part.labels for t in part.tokens[lbl] if t.id is not None})
        if not ids:
            raise BackendError("named-token readout needs verified token ids; run the probe")
        body["logprobs"] = True
        body["top_logprobs"] = 1
        body["logprob_token_ids"] = ids
        body["return_as_token_id"] = True

    @staticmethod
    def read(parsed: Parsed, part: RenderedPart) -> Readout:
        readout = read_answer_set(parsed, part, full_softmax=True)
        if readout.missing:
            raise BackendError(
                f"labels {list(readout.missing)} missing from the named-token logprobs; "
                "the backend may have ignored logprob_token_ids (re-run the probe)"
            )
        return readout


def _answer_texts(part: RenderedPart) -> list[str]:
    return list(dict.fromkeys(t.text for lbl in part.labels for t in part.tokens[lbl]))


def _answer_ids(part: RenderedPart) -> list[int]:
    return sorted({t.id for lbl in part.labels for t in part.tokens[lbl] if t.id is not None})


class GrammarRung:
    """Rung 2: a grammar masks the vocabulary to the answer spellings and the server reports
    post-sampling probabilities renormalized over the survivors (llama.cpp)."""

    rung = Rung.GRAMMAR

    @staticmethod
    def available(capabilities: Capabilities | None) -> bool:
        return bool(capabilities and Rung.GRAMMAR in capabilities.rungs)

    @staticmethod
    def shape(body: dict[str, Any], part: RenderedPart, top_k: int, dialect: Dialect) -> None:
        texts = _answer_texts(part)
        body["grammar"] = grammar_for(texts)
        neutral_samplers(body)
        n = max(top_k, len(texts) + 4)
        body["n_probs" if "n_predict" in body else "logprobs"] = n
        if "messages" in body:
            body["top_logprobs"] = n

    FLOOR_LOGPROB = -40.0  # below anything a float32 softmax reports before underflow

    @staticmethod
    def read(parsed: Parsed, part: RenderedPart) -> Readout:
        readout = read_answer_set(parsed, part, full_softmax=False)
        if not readout.missing:
            return readout
        if not readout.logprobs:
            raise BackendError(
                "no answer label returned under the grammar; "
                "the server may not honour grammar with post-sampling probabilities"
            )
        # Under a mask the reported list holds every answer with non-zero mass, so a label
        # that is absent underflowed to zero: a certain answer, floored, not a degraded one.
        logprobs = dict(readout.logprobs)
        for label in readout.missing:
            logprobs[label] = GrammarRung.FLOOR_LOGPROB
        return Readout(logprobs, None, readout.missing, degraded=False)


class EqualBiasRung:
    """Rung 4: the same bias on every answer token lifts them into the reported list without
    changing their ratios (softmax is shift-invariant within the biased set)."""

    rung = Rung.EQUAL_BIAS
    BIAS = 50

    @staticmethod
    def available(capabilities: Capabilities | None) -> bool:
        return bool(capabilities and Rung.EQUAL_BIAS in capabilities.rungs)

    @staticmethod
    def shape(body: dict[str, Any], part: RenderedPart, top_k: int, dialect: Dialect) -> None:
        ids = _answer_ids(part)
        keys = [str(i) for i in ids] if ids else _answer_texts(part)
        body["logit_bias"] = {k: EqualBiasRung.BIAS for k in keys}
        if dialect is Dialect.LLAMACPP:
            neutral_samplers(body)
        n = max(top_k, len(keys) + 4)
        if "messages" in body:
            body["top_logprobs"] = n
        else:
            body["n_probs" if "n_predict" in body else "logprobs"] = n

    @staticmethod
    def read(parsed: Parsed, part: RenderedPart) -> Readout:
        readout = read_answer_set(parsed, part, full_softmax=False)
        if readout.missing:
            raise BackendError(
                f"labels {list(readout.missing)} absent after equal bias; "
                "the server may report logprobs before bias is applied"
            )
        return readout


class TopKFloorRung:
    """Rung 5: top-k, but a label that fell out of the list gets the observed floor minus a
    margin, and the answer is marked degraded with the missing labels named."""

    rung = Rung.TOP_K_FLOOR
    MARGIN = 1.0

    @staticmethod
    def available(capabilities: Capabilities | None) -> bool:
        return True

    @staticmethod
    def shape(body: dict[str, Any], part: RenderedPart, top_k: int, dialect: Dialect) -> None:
        TopKRung.shape(body, part, top_k, dialect)

    @staticmethod
    def read(parsed: Parsed, part: RenderedPart) -> Readout:
        readout = read_answer_set(parsed, part, full_softmax=True)
        if not readout.missing:
            return readout
        if not parsed.entries:
            raise BackendError("no logprobs returned; nothing to floor against")
        floor = min(e.logprob for e in parsed.entries) - TopKFloorRung.MARGIN
        logprobs = dict(readout.logprobs)
        for label in readout.missing:
            logprobs[label] = floor
        return Readout(logprobs, None, readout.missing, degraded=True)


RUNGS: dict[Rung, Any] = {
    Rung.TOP_K_FLOOR: TopKFloorRung,
    Rung.TOP_K: TopKRung,
    Rung.NAMED_TOKEN_LOGPROBS: NamedTokenLogprobsRung,
    Rung.GRAMMAR: GrammarRung,
    Rung.EQUAL_BIAS: EqualBiasRung,
}
