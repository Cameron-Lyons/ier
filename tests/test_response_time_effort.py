"""Response time effort compares every response time with its own item threshold."""

from decimal import Decimal
from fractions import Fraction
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ier import response_time_effort, response_time_effort_flag

# NT10 thresholds: 10% of the item means 20, 40, 40/3 (one missing time), and 60.
_TIMES = np.array(
    [
        [20.0, 40.0, 8.0, 60.0],
        [1.0, 50.0, 12.0, 30.0],
        [30.0, 3.0, np.nan, 90.0],
        [29.0, 67.0, 20.0, 60.0],
    ]
)


def _reference(times: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count rapid answers one respondent and item at a time."""
    scores = np.full(len(times), np.nan)
    rapid = np.zeros(times.shape, dtype=bool)
    for row, values in enumerate(times):
        answered = 0
        effortful = 0
        for item, value in enumerate(values):
            threshold = thresholds[item]
            if np.isnan(value) or not np.isfinite(threshold) or threshold <= 0:
                continue
            answered += 1
            if value < threshold:
                rapid[row, item] = True
            else:
                effortful += 1
        if answered:
            scores[row] = effortful / answered
    return scores, rapid


def test_normative_thresholds_count_rapid_answers_among_answered_items() -> None:
    scores, rapid = response_time_effort(_TIMES, return_item_flags=True)

    assert scores.tolist() == [1.0, 0.75, 2 / 3, 1.0]
    assert rapid.tolist() == [
        [False, False, False, False],
        [True, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
    ]
    assert rapid.dtype == bool
    np.testing.assert_array_equal(response_time_effort(_TIMES), scores)


def test_times_equal_to_the_threshold_are_not_rapid() -> None:
    times = np.array([[1.0, 4.0], [0.5, 3.999]])
    assert response_time_effort(times, thresholds=[1.0, 4.0]).tolist() == [1.0, 0.0]


def test_normative_fraction_scales_the_item_means() -> None:
    scores = response_time_effort(_TIMES, normative_fraction=0.5)
    thresholds = 0.5 * np.array([20.0, 40.0, 40 / 3, 60.0])
    np.testing.assert_array_equal(scores, _reference(_TIMES, thresholds)[0])
    # Thirty seconds equals the doubled third threshold and is not rapid.
    assert scores.tolist() == [1.0, 0.75, 2 / 3, 1.0]


def test_maximum_threshold_caps_normative_and_explicit_thresholds() -> None:
    times = np.array([[15.0, 2.0], [250.0, 3.0], [235.0, 1.0]])
    # The first item's NT10 threshold is 50 / 3; a cap of 10 keeps 15 seconds effortful.
    assert response_time_effort(times).tolist() == [0.5, 1.0, 1.0]
    assert response_time_effort(times, max_threshold=10).tolist() == [1.0, 1.0, 1.0]
    assert response_time_effort(times, [20.0, 2.5], max_threshold=Fraction(3)).tolist() == [
        0.5,
        1.0,
        0.5,
    ]


def test_shared_and_per_item_thresholds_are_supported() -> None:
    assert response_time_effort(_TIMES, thresholds=25).tolist() == [0.5, 0.5, 2 / 3, 0.75]
    assert response_time_effort(_TIMES, thresholds=np.float32(25)).tolist() == [
        0.5,
        0.5,
        2 / 3,
        0.75,
    ]
    per_item = np.array([2.0, 4.0, np.nan, 0.0])
    scores, rapid = response_time_effort(_TIMES, per_item, return_item_flags=True)
    # Unavailable and nonpositive thresholds exclude their items from every denominator.
    assert scores.tolist() == [1.0, 0.5, 0.5, 1.0]
    assert not rapid[:, 2:].any()
    negative = response_time_effort(-_TIMES, thresholds=[-1.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_equal(negative, [0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("infinity", [np.inf, -np.inf, np.float32(np.inf), np.longdouble(np.inf)])
def test_explicit_infinite_thresholds_exclude_their_items(infinity: Any) -> None:
    per_item = np.array([2.0, 4.0, 1.0, 5.0], dtype=np.asarray(infinity).dtype)
    excluded = per_item.copy()
    excluded[2:] = [np.nan, 0.0]
    infinite = excluded.copy()
    infinite[2] = infinity
    expected, expected_rapid = response_time_effort(_TIMES, excluded, return_item_flags=True)

    scores, rapid = response_time_effort(_TIMES, infinite, return_item_flags=True)

    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(rapid, expected_rapid)
    np.testing.assert_array_equal(scores, _reference(_TIMES, infinite.astype(float))[0])
    assert np.isnan(response_time_effort(_TIMES, thresholds=infinity)).all()
    _, flags = response_time_effort_flag(_TIMES, thresholds=[infinity] * 4)
    assert not flags.any()


def test_maximum_threshold_caps_infinite_thresholds() -> None:
    capped = response_time_effort(_TIMES, [np.inf, 4.0, np.nan, -np.inf], max_threshold=25.0)
    np.testing.assert_array_equal(capped, response_time_effort(_TIMES, [25.0, 4.0, np.nan, -1.0]))
    # An infinite time makes the second normative threshold infinite; both are capped at 10.
    times = np.array([[12.0, np.inf], [30.0, 1.0]])
    normative = response_time_effort(times, max_threshold=10.0)
    explicit = response_time_effort(times, [2.0, np.inf], max_threshold=10.0)
    assert normative.tolist() == explicit.tolist() == [1.0, 0.5]


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="long double has no values beyond the float64 range",
)
def test_finite_thresholds_beyond_float64_are_still_rejected() -> None:
    thresholds = np.array([np.inf, 1.0, 1.0, 1.0], dtype=np.longdouble)
    thresholds[1] = np.longdouble("1e400")
    with pytest.raises(ValueError, match="thresholds must contain only finite values or NaN"):
        response_time_effort(_TIMES, thresholds)


def test_unanswered_rows_and_items_are_unavailable() -> None:
    times = np.array(
        [
            [np.nan, np.nan, 4.0],
            [np.nan, np.nan, np.nan],
            [2.0, np.nan, 0.1],
            [5.0, np.nan, np.nan],
        ]
    )
    scores, rapid = response_time_effort(times, return_item_flags=True)
    # The second item has no times and therefore no normative threshold.
    np.testing.assert_array_equal(scores, [1.0, np.nan, 0.5, 1.0])
    assert rapid.tolist() == [
        [False, False, False],
        [False, False, False],
        [False, False, True],
        [False, False, False],
    ]

    only_excluded = response_time_effort(times, thresholds=[np.nan, 1.0, np.nan])
    assert np.isnan(only_excluded).all()


def test_integer_and_single_precision_times_match_double_precision() -> None:
    rng = np.random.default_rng(4)
    seconds = rng.integers(1, 60, size=(300, 12))
    expected, expected_rapid = response_time_effort(seconds.astype(float), return_item_flags=True)

    for times in (seconds, seconds.astype(np.int32), seconds.astype(np.float32)):
        scores, rapid = response_time_effort(times, return_item_flags=True)
        np.testing.assert_array_equal(scores, expected)
        np.testing.assert_array_equal(rapid, expected_rapid)
    np.testing.assert_array_equal(response_time_effort(seconds.tolist()), expected)
    np.testing.assert_array_equal(response_time_effort(np.asfortranarray(seconds)), expected)


def test_single_precision_times_compare_with_double_precision_thresholds() -> None:
    times = np.array([[0.1, 1.0], [0.1, 1.0]], dtype=np.float32)
    # float32(0.1) exceeds the double threshold 0.1, so it is not rapid.
    assert response_time_effort(times, thresholds=0.1).tolist() == [1.0, 1.0]
    assert response_time_effort(times, thresholds=float(np.float32(0.1)) + 1e-12).tolist() == [
        0.5,
        0.5,
    ]


def test_random_matrices_match_the_itemwise_reference() -> None:
    rng = np.random.default_rng(12)
    times = rng.lognormal(mean=1.0, sigma=1.0, size=(500, 9))
    times[rng.random(times.shape) < 0.2] = np.nan
    times[:5] = np.nan
    times[:, 3] = np.nan
    answered = ~np.isnan(times)
    with np.errstate(invalid="ignore"):
        means = np.sum(times, axis=0, where=answered) / np.sum(answered, axis=0)
    expected = _reference(times, np.minimum(0.15 * means, 2.0))

    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 9 * 7):
        scores, rapid = response_time_effort(
            times, normative_fraction=0.15, max_threshold=2.0, return_item_flags=True
        )

    np.testing.assert_array_equal(scores, expected[0])
    np.testing.assert_array_equal(rapid, expected[1])
    assert np.isnan(scores[:5]).all()


def test_masked_times_are_unanswered() -> None:
    raw = np.array([[1.0, -99.0, 30.0], [20.0, 2.0, 40.0], [0.5, 3.0, -99.0]])
    masked = np.ma.masked_equal(raw, -99.0)
    nan_coded = np.where(raw == -99.0, np.nan, raw)
    np.testing.assert_array_equal(response_time_effort(masked), response_time_effort(nan_coded))


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"normative_fraction": 0}, "normative_fraction must be greater than 0"),
        ({"normative_fraction": -0.1}, "normative_fraction must be between 0 and 1"),
        ({"normative_fraction": 1.5}, "normative_fraction must be between 0 and 1"),
        ({"normative_fraction": True}, "normative_fraction must be between 0 and 1"),
        ({"normative_fraction": "0.1"}, "normative_fraction must be between 0 and 1"),
        ({"max_threshold": 0}, "max_threshold must be a positive finite number"),
        ({"max_threshold": -1.0}, "max_threshold must be a positive finite number"),
        ({"max_threshold": np.inf}, "max_threshold must be a positive finite number"),
        ({"max_threshold": np.nan}, "max_threshold must be a positive finite number"),
        ({"max_threshold": True}, "max_threshold must be a positive finite number"),
        ({"max_threshold": "10"}, "max_threshold must be a positive finite number"),
        ({"max_threshold": [10]}, "max_threshold must be a positive finite number"),
        ({"max_threshold": Decimal("1e999")}, "max_threshold must be a positive finite number"),
        ({"max_threshold": Decimal("sNaN")}, "max_threshold must be a positive finite number"),
        ({"max_threshold": Fraction(10**400)}, "max_threshold must be a positive finite number"),
        ({"thresholds": [1.0, 2.0]}, r"one value per item \(4\), got 2"),
        ({"thresholds": [[1.0] * 4]}, "thresholds must be one-dimensional"),
        ({"thresholds": [True] * 4}, "thresholds must be a one-dimensional real numeric"),
        ({"thresholds": True}, "thresholds must be a one-dimensional real numeric"),
        ({"thresholds": "1.0"}, "thresholds must be a one-dimensional real numeric"),
        ({"thresholds": [[1.0, np.inf, 1.0, 1.0]]}, "thresholds must be one-dimensional"),
        ({"thresholds": np.array([1, 2, 3, 4], dtype="m8[s]")}, "one-dimensional real numeric"),
        ({"thresholds": []}, "thresholds cannot be empty"),
        ({"thresholds": [1.0, [2.0], 3.0, 4.0]}, "thresholds must be a real number or"),
        ({"return_item_flags": 1}, "return_item_flags must be a boolean"),
    ],
)
def test_invalid_options_are_rejected(keywords: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        response_time_effort(_TIMES, **keywords)


def test_positive_numeric_caps_are_accepted() -> None:
    expected = response_time_effort(_TIMES, max_threshold=1.5)
    for cap in (np.float32(1.5), np.array(1.5), Decimal("1.5"), Fraction(3, 2)):
        np.testing.assert_array_equal(response_time_effort(_TIMES, max_threshold=cap), expected)
    np.testing.assert_array_equal(
        response_time_effort(_TIMES, normative_fraction=np.float64(0.1)),
        response_time_effort(_TIMES),
    )


def test_flags_mark_scores_strictly_below_the_cutoff() -> None:
    times = np.vstack([_TIMES, np.full(4, np.nan)])
    scores, flags = response_time_effort_flag(times)
    np.testing.assert_array_equal(scores, response_time_effort(times))
    assert flags.tolist() == [False, True, True, False, False]

    _, at_cutoff = response_time_effort_flag(times, threshold=0.75)
    assert at_cutoff.tolist() == [False, False, True, False, False]
    _, capped = response_time_effort_flag(times, 0.9, thresholds=1.5, max_threshold=1.0)
    assert capped.tolist() == [False, False, False, False, False]
    _, below_mean = response_time_effort_flag(times, threshold=1, normative_fraction=1.0)
    assert below_mean.tolist() == [True, True, True, False, False]


@pytest.mark.parametrize("threshold", [1.5, -0.1, np.nan, True, "0.9", None])
def test_flag_cutoff_must_be_a_proportion(threshold: Any) -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        response_time_effort_flag(_TIMES, threshold=threshold)
