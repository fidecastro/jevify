"""Metrics (ADR-0005 D4), standard library only. Every function takes plain lists so the
expected values in tests are literals, never a recomputation of the same code."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

_EPSILON = 1e-12


def accuracy(selected: Sequence[str], labels: Sequence[str]) -> float:
    if not labels:
        return float("nan")
    return sum(s == label for s, label in zip(selected, labels, strict=True)) / len(labels)


def log_loss(probabilities: Sequence[Mapping[str, float]], labels: Sequence[str]) -> float:
    total = 0.0
    for probs, label in zip(probabilities, labels, strict=True):
        total -= math.log(max(probs.get(label, 0.0), _EPSILON))
    return total / len(labels)


def brier_score(probabilities: Sequence[Mapping[str, float]], labels: Sequence[str]) -> float:
    total = 0.0
    for probs, label in zip(probabilities, labels, strict=True):
        total += sum((p - (1.0 if key == label else 0.0)) ** 2 for key, p in probs.items())
    return total / len(labels)


def expected_calibration_error(
    probabilities: Sequence[Mapping[str, float]], labels: Sequence[str], *, bins: int = 10
) -> tuple[float, list[dict[str, float | int]]]:
    """ECE over equal-width confidence bins on the top probability, plus the reliability table.

    Confidence 1.0 falls in the last bin. Empty bins contribute nothing.
    """
    table = [
        {"lower": i / bins, "upper": (i + 1) / bins, "count": 0, "confidence": 0.0, "accuracy": 0.0}
        for i in range(bins)
    ]
    for probs, label in zip(probabilities, labels, strict=True):
        top_label = max(probs, key=lambda k: probs[k])
        confidence = probs[top_label]
        index = min(int(confidence * bins), bins - 1)
        cell = table[index]
        cell["count"] += 1
        cell["confidence"] += confidence
        cell["accuracy"] += 1.0 if top_label == label else 0.0
    total = len(labels)
    ece = 0.0
    for cell in table:
        if cell["count"]:
            cell["confidence"] /= cell["count"]
            cell["accuracy"] /= cell["count"]
            ece += cell["count"] / total * abs(cell["accuracy"] - cell["confidence"])
    return ece, table


def mean_absolute_error(expectations: Sequence[float], labels: Sequence[int]) -> float:
    return sum(abs(e - label) for e, label in zip(expectations, labels, strict=True)) / len(labels)


def ranked_probability_score(
    distributions: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    """Mean RPS over ordered levels: squared distance between cumulative distributions,
    divided by the number of levels minus one."""
    total = 0.0
    for probs, label in zip(distributions, labels, strict=True):
        k = len(probs)
        cdf_p = 0.0
        cdf_t = 0.0
        score = 0.0
        for index, p in enumerate(probs):
            cdf_p += p
            cdf_t += 1.0 if index == label else 0.0
            score += (cdf_p - cdf_t) ** 2
        total += score / max(1, k - 1)
    return total / len(labels)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
