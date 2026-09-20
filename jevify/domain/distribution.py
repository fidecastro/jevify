"""The statistics chokepoint.

Every probability an answer carries, every confidence and every score
expectation is computed here and nowhere else (a structure guard enforces
that no other module defines `confidence`). The definitions match the
score-semantics contract in docs/00-invariants.md §5:

- a distribution is a softmax over the answer set, so it sums to one;
- confidence for choice and score is one minus the normalized Shannon entropy;
- confidence for noul is the distance of P(true) from one half, scaled to [0, 1];
- the score expectation is the probability-weighted level index.

    >>> d = Distribution.from_logprobs({"a": -1.0, "b": -2.0})
    >>> round(d.probabilities[0], 4)
    0.7311
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

QuestionKind = Literal["choice", "score", "noul"]

_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Distribution:
    """A normalized distribution over an ordered answer set."""

    labels: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("a distribution needs at least one label")
        if len(self.labels) != len(self.probabilities):
            raise ValueError("labels and probabilities must have the same length")
        total = sum(self.probabilities)
        if abs(total - 1.0) > _TOLERANCE:
            raise ValueError(f"probabilities must sum to 1, got {total!r}")

    @classmethod
    def from_logprobs(cls, logprobs: Mapping[str, float]) -> Distribution:
        """Softmax over the given log-probabilities, in the mapping's order.

        Shift-invariant by construction: only differences between values matter.
        """
        if not logprobs:
            raise ValueError("a distribution needs at least one label")
        labels = tuple(logprobs)
        values = [logprobs[label] for label in labels]
        peak = max(values)
        weights = [math.exp(value - peak) for value in values]
        total = sum(weights)
        return cls(labels, tuple(weight / total for weight in weights))

    @classmethod
    def from_token_logprobs(cls, token_logprobs: Mapping[str, Mapping[str, float]]) -> Distribution:
        """Build from per-label token sets: a label's mass is the sum over its tokens.

        `token_logprobs` maps label -> {token: logprob}. Tokens are distinct
        events (different spellings at the same answer slot), so their
        probabilities add.
        """
        if not token_logprobs:
            raise ValueError("a distribution needs at least one label")
        masses = {
            label: sum(math.exp(lp) for lp in tokens.values())
            for label, tokens in token_logprobs.items()
        }
        total = sum(masses.values())
        if total <= 0.0:
            raise ValueError("no probability mass on any label token")
        return cls(tuple(masses), tuple(mass / total for mass in masses.values()))

    def confidence(self, kind: QuestionKind) -> float:
        """A statistic of this distribution, never an estimate that the decision is right."""
        if kind == "noul":
            return min(1.0, max(0.0, abs(self.probabilities[-1] - 0.5) * 2.0))
        k = len(self.probabilities)
        if k < 2:
            return 1.0
        entropy = -sum(p * math.log(p) for p in self.probabilities if p > 0.0)
        return min(1.0, max(0.0, 1.0 - entropy / math.log(k)))

    def expectation(self) -> float:
        """Probability-weighted level index: the Jev `score` value."""
        return sum(index * p for index, p in enumerate(self.probabilities))

    def argmax(self) -> str:
        """The most probable label; the first one on ties."""
        best = 0
        for index, p in enumerate(self.probabilities):
            if p > self.probabilities[best]:
                best = index
        return self.labels[best]

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.labels, self.probabilities, strict=True))
