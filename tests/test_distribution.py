"""The statistics chokepoint: every probability, confidence and expectation comes from here."""

from __future__ import annotations

import math

import pytest

from jevify.domain.distribution import Distribution

LN = math.log


def test_from_logprobs_normalizes_over_answer_set() -> None:
    d = Distribution.from_logprobs({"a": -1.0, "b": -2.0})
    # e^-1 / (e^-1 + e^-2) = 1 / (1 + e^-1)
    assert d.labels == ("a", "b")
    assert d.probabilities[0] == pytest.approx(0.7311, abs=1e-4)
    assert d.probabilities[1] == pytest.approx(0.2689, abs=1e-4)
    assert sum(d.probabilities) == pytest.approx(1.0)


def test_from_logprobs_is_shift_invariant() -> None:
    a = Distribution.from_logprobs({"x": -100.0, "y": -101.0})
    b = Distribution.from_logprobs({"x": 3.0, "y": 2.0})
    assert a.probabilities == pytest.approx(b.probabilities)


def test_label_mass_sums_over_token_set() -> None:
    # "no" is spelled two ways at the answer slot; both belong to the same label.
    d = Distribution.from_token_logprobs(
        {"no": {"no": LN(0.6), "No": LN(0.2)}, "yes": {"yes": LN(0.1)}}
    )
    assert d.labels == ("no", "yes")
    assert d.probabilities[0] == pytest.approx(8 / 9)
    assert d.probabilities[1] == pytest.approx(1 / 9)


def test_choice_confidence_is_jevs_peak_statistic() -> None:
    """Jev: confidence = (n * peak - 1) / (n - 1); 0 when uniform, 1 when certain."""
    assert Distribution(("a", "b"), (0.5, 0.5)).confidence("choice") == pytest.approx(0.0)
    assert Distribution(("a", "b"), (1.0, 0.0)).confidence("choice") == pytest.approx(1.0)
    uniform4 = Distribution(("a", "b", "c", "d"), (0.25,) * 4)
    assert uniform4.confidence("choice") == pytest.approx(0.0)
    # TypeSafe's worked example: [0.90, 0.06, 0.04] -> (3 * 0.90 - 1) / 2 = 0.85
    documented = Distribution(("a", "b", "c"), (0.90, 0.06, 0.04))
    assert documented.confidence("choice") == pytest.approx(0.85)
    # Two options: (2 * 0.9 - 1) / 1 = 0.8
    skewed = Distribution(("a", "b"), (0.9, 0.1))
    assert skewed.confidence("choice") == pytest.approx(0.8)


def test_score_confidence_uses_the_same_definition_as_choice() -> None:
    d = Distribution(("0", "1", "2"), (0.05, 0.86, 0.09))
    assert d.confidence("score") == pytest.approx(d.confidence("choice"))
    assert d.confidence("score") == pytest.approx((3 * 0.86 - 1) / 2)


def test_noul_confidence_is_distance_from_half() -> None:
    assert Distribution(("false", "true"), (0.1, 0.9)).confidence("noul") == pytest.approx(0.8)
    assert Distribution(("false", "true"), (0.5, 0.5)).confidence("noul") == pytest.approx(0.0)


def test_score_expectation_uses_level_index() -> None:
    d = Distribution(("calm", "civil", "angry"), (0.05, 0.86, 0.09))
    assert d.expectation() == pytest.approx(1.04)


def test_argmax_returns_first_label_on_ties() -> None:
    assert Distribution(("a", "b"), (0.5, 0.5)).argmax() == "a"
    assert Distribution(("a", "b"), (0.2, 0.8)).argmax() == "b"


def test_single_label_is_certain() -> None:
    d = Distribution.from_logprobs({"only": -3.0})
    assert d.probabilities == (1.0,)
    assert d.confidence("choice") == 1.0


def test_rejects_empty_and_unnormalized() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        Distribution.from_logprobs({})
    with pytest.raises(ValueError, match="sum to 1"):
        Distribution(("a", "b"), (0.6, 0.6))
    with pytest.raises(ValueError, match="same length"):
        Distribution(("a", "b"), (1.0,))
