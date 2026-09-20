"""The readout ladder (ADR-0002 D4): how a distribution over the answer set is obtained.

Each rung knows whether it is available under a capability list, how to
shape the request, and how to read the answer tokens out of the response.
The token match is id-first, text second (ADR-0002 D2, plan decision 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from jevify.adapters.endpoint.dialects import Entry, Parsed
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


RUNGS: dict[Rung, type[TopKRung] | type[NamedTokenLogprobsRung]] = {
    Rung.TOP_K: TopKRung,
    Rung.NAMED_TOKEN_LOGPROBS: NamedTokenLogprobsRung,
}
