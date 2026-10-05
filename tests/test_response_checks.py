"""Regression coverage for bounded missing-response and attention-check scoring."""

import math
from unittest.mock import patch

import numpy as np
import pytest

import ier._row_statistics as row_statistics
from ier import infrequency, infrequency_flag, missing_rate
from ier.types import InfrequencyMissingPolicy


def _responses(layout: str, dtype: type) -> np.ndarray:
    rng = np.random.default_rng(19)
    data = rng.integers(1, 6, size=(17, 9)).astype(dtype)
    data[rng.random(data.shape) < 0.3] = np.nan
    data[0] = np.nan
    data[1] = 3
    data[2, 2] = np.inf
    data[3, 4] = -np.inf
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "strided":
        data = data[::-2, ::-1]
    data.flags.writeable = False
    return data


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("proportion", [False, True])
@pytest.mark.parametrize("policy", ["pass", "fail", "omit", "propagate"])
def test_attention_checks_match_scalar_reference(
    layout: str, dtype: type, proportion: bool, policy: InfrequencyMissingPolicy
) -> None:
    data = _responses(layout, dtype)
    indices = [8, 2, 4, 0]
    expected = [5.0, 1.0, 3.0, 2.0]
    original = data.copy()
    reference = []
    for row in data:
        checks = [(row[index], answer) for index, answer in zip(indices, expected, strict=True)]
        observed = [(value, answer) for value, answer in checks if not np.isnan(value)]
        if (policy == "omit" and not observed) or (
            policy == "propagate" and len(observed) != len(checks)
        ):
            reference.append(np.nan)
            continue
        failures = sum(value != answer for value, answer in observed)
        if policy == "fail":
            failures += len(checks) - len(observed)
        denominator = len(observed) if policy == "omit" else len(checks)
        reference.append(failures / denominator if proportion else float(failures))

    # Exercise full batches, the final partial batch, and rows wider than the budget.
    for budget in (3, 13):
        with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", budget):
            actual = infrequency(data, indices, expected, proportion=proportion, missing=policy)
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("subset", [None, [8, 2, 4, 0]])
@pytest.mark.parametrize("use_mask", [False, True])
def test_missing_rates_match_scalar_reference(
    layout: str, subset: list[int] | None, use_mask: bool
) -> None:
    data = _responses(layout, np.float64)
    indices = list(range(data.shape[1])) if subset is None else subset
    rng = np.random.default_rng(21)
    mask = rng.random(data.shape) < 0.8
    mask[0] = False
    mask[1] = True
    mask.flags.writeable = False
    original_mask = mask.copy()
    reference = []
    for row, applicable in zip(data, mask, strict=True):
        values = [row[index] for index in indices if not use_mask or applicable[index]]
        reference.append(
            sum(np.isnan(value) for value in values) / len(values) if values else np.nan
        )

    for budget in (3, 23):
        with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", budget):
            actual = missing_rate(data, subset, applicable_mask=mask if use_mask else None)
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(mask, original_mask)


def test_checks_preserve_numeric_input_handling() -> None:
    integers = np.arange(120, dtype=np.int64).reshape(15, 8)
    indices = [np.int64(7), np.int64(2)]
    with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 5):
        np.testing.assert_array_equal(missing_rate(integers, indices), np.zeros(15))
        np.testing.assert_array_equal(infrequency(integers, indices, [7, 2]), [0] + [2] * 14)
        np.testing.assert_array_equal(missing_rate([["1", "nan"], ["nan", "nan"]]), [0.5, 1])


@pytest.mark.parametrize(
    "dtype,base", [(np.int64, -(2**63)), (np.int64, 2**60), (np.uint64, 2**64 - 3)]
)
@pytest.mark.parametrize("policy", ["pass", "fail", "omit", "propagate"])
@pytest.mark.parametrize("proportion", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_attention_checks_preserve_exact_integer_categories(
    dtype: type, base: int, policy: InfrequencyMissingPolicy, proportion: bool, layout: str
) -> None:
    answers = [base + 1, np.array(base + 2, dtype=dtype)[()], 0.5, -1, 2**64, float(base)]
    data = np.repeat(np.array([[base], [base + 1], [base + 2]], dtype=dtype), len(answers), axis=1)
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "strided":
        data = np.repeat(data, 2, axis=1)[:, ::2]
    original = data.copy()
    data.flags.writeable = False
    reference = [
        sum(
            value.item() != (answer.item() if isinstance(answer, np.generic) else answer)
            for value, answer in zip(row, answers, strict=True)
        )
        for row in data
    ]
    if proportion:
        reference = [count / len(answers) for count in reference]
    with (
        patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 7),
        np.errstate(all="raise"),
    ):
        actual = infrequency(
            data, list(range(len(answers))), answers, proportion=proportion, missing=policy
        )
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("policy", ["pass", "fail", "omit", "propagate"])
def test_unrepresentable_integer_answers_do_not_match_float_neighbours(
    dtype: type, policy: InfrequencyMissingPolicy
) -> None:
    base = 2**60
    data = np.array([[base, base], [np.nan, base]], dtype=dtype)
    scores, flags = infrequency_flag(data, [0, 1], [base + 1, np.uint64(base + 2)], missing=policy)
    expected = [2, np.nan if policy == "propagate" else 2 if policy == "fail" else 1]
    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(flags, [True, policy != "propagate"])


@pytest.mark.parametrize("policy", ["pass", "fail", "omit", "propagate"])
@pytest.mark.parametrize("dtype", [bool, np.int8, np.int64, np.uint64])
def test_integer_attention_checks_skip_missing_scan(
    policy: InfrequencyMissingPolicy, dtype: type
) -> None:
    data = np.array([[False, True], [True, False]], dtype=dtype)
    with patch("ier.infrequency.np.isnan", side_effect=AssertionError("integers have no NaN")):
        actual = infrequency(data, [0, 1], [0, 1], proportion=True, missing=policy)
    np.testing.assert_array_equal(actual, [0, 1])


@pytest.mark.parametrize(
    "invalid",
    [[2**1024], ["not-numeric"], [np.inf], [1 + 1j], np.array([1 + 0j]), np.array([1 + 1j])],
)
def test_invalid_attention_answers_raise_clear_errors(invalid: list) -> None:
    with pytest.raises(ValueError, match="finite numeric"):
        infrequency([[1]], [0], invalid)


@pytest.mark.parametrize("answers", [["0", "1"], (0.0, 1.0), np.array([False, True])])
def test_attention_answer_conversion_keeps_legacy_numeric_forms(answers: list) -> None:
    np.testing.assert_array_equal(infrequency([[0, 1], [1, 0]], [0, 1], answers), [0, 2])


@pytest.mark.parametrize("policy", ["pass", "fail", "omit", "propagate"])
def test_attention_check_workspaces_are_bounded(policy: InfrequencyMissingPolicy) -> None:
    data = np.ones((50, 8))
    with (
        patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 20),
        patch("ier.infrequency.np.sum", wraps=np.sum) as reductions,
    ):
        infrequency(data, [6, 2, 0], [1, 1, 1], missing=policy)
    assert reductions.call_count > 1
    assert all(call.args[0].size <= 20 for call in reductions.call_args_list)


@pytest.mark.parametrize("subset", [None, [6, 2, 0]])
@pytest.mark.parametrize("use_mask", [False, True])
def test_missing_rate_workspaces_are_bounded(subset: list[int] | None, use_mask: bool) -> None:
    data = np.ones((50, 8))
    mask = np.ones(data.shape, dtype=bool) if use_mask else None
    with (
        patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", 20),
        patch("ier.missing.np.isnan", wraps=np.isnan) as missing_checks,
    ):
        missing_rate(data, subset, mask)
    assert missing_checks.call_count > 1
    assert all(call.args[0].size <= 20 for call in missing_checks.call_args_list)


def test_acceptable_ranges_accept_every_answer_inside_the_range() -> None:
    # A bogus item accepts either disagreement category; agreement fails.
    responses = [[1], [2], [3], [4], [5]]
    np.testing.assert_array_equal(
        infrequency(responses, [0], acceptable_ranges=[(1, 2)]), [0, 0, 1, 1, 1]
    )
    np.testing.assert_array_equal(infrequency(responses, [0], [1]), [0, 1, 1, 1, 1])
    scores, flags = infrequency_flag(responses, [0], acceptable_ranges=[(1, 2)])
    np.testing.assert_array_equal(scores, [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(flags, [False, False, True, True, True])


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64, np.float32, np.float64])
def test_open_acceptable_ranges(dtype: type) -> None:
    data = np.array([[1, 7, 4], [4, 3, 7], [7, 1, 1]], dtype=dtype)
    ranges = [(-np.inf, 2), (4, float("inf")), (-math.inf, math.inf)]
    np.testing.assert_array_equal(
        infrequency(data, [0, 1, 2], acceptable_ranges=ranges), [0.0, 2.0, 2.0]
    )


@pytest.mark.parametrize("policy", ["pass", "fail", "omit", "propagate"])
@pytest.mark.parametrize("proportion", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_acceptable_ranges_match_scalar_reference(
    policy: InfrequencyMissingPolicy, proportion: bool, layout: str, dtype: type
) -> None:
    data = _responses(layout, dtype)
    indices = [8, 2, 4, 0]
    ranges = [(4.0, 5.0), (-math.inf, 1.5), (2, 3), (3, math.inf)]
    original = data.copy()
    reference = []
    for row in data:
        checks = [(row[index], bounds) for index, bounds in zip(indices, ranges, strict=True)]
        observed = [(value, bounds) for value, bounds in checks if not np.isnan(value)]
        if (policy == "omit" and not observed) or (
            policy == "propagate" and len(observed) != len(checks)
        ):
            reference.append(np.nan)
            continue
        failures = sum(not low <= value <= high for value, (low, high) in observed)
        if policy == "fail":
            failures += len(checks) - len(observed)
        denominator = len(observed) if policy == "omit" else len(checks)
        reference.append(failures / denominator if proportion else float(failures))

    for budget in (3, 13):
        with patch.object(row_statistics, "_ROW_BATCH_ELEMENTS", budget):
            actual = infrequency(
                data, indices, proportion=proportion, missing=policy, acceptable_ranges=ranges
            )
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_integer_ranges_stay_exact_at_dtype_extremes(dtype: type, layout: str) -> None:
    bounds = np.iinfo(dtype)
    low, high = int(bounds.min), int(bounds.max)
    data = np.array([[low, low + 1, high - 1, high]] * 4, dtype=dtype).T.copy()
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "strided":
        data = np.repeat(data, 2, axis=1)[:, ::2]
    data.flags.writeable = False
    ranges = [
        (low + 1, high),  # Float conversion would round both ends to the extremes.
        (float(high), math.inf),  # 2**63 or 2**64 is above every category.
        (-(2**70), low),  # Huge integer bounds clamp to the dtype.
        (high - 1, 2**70),
    ]
    with np.errstate(all="raise"):
        actual = infrequency(data, [0, 1, 2, 3], acceptable_ranges=ranges)
    reference = [
        sum(
            not lower <= value <= upper
            for value, (lower, upper) in zip(row.tolist(), ranges, strict=True)
        )
        for row in data
    ]
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(actual, [3, 3, 2, 2])


@pytest.mark.parametrize("dtype", [bool, np.int8, np.int64, np.uint64])
@pytest.mark.parametrize("policy", ["pass", "fail", "omit", "propagate"])
def test_empty_integer_ranges_fail_every_response(
    dtype: type, policy: InfrequencyMissingPolicy
) -> None:
    data = np.array([[False, True], [True, False]], dtype=dtype)
    ranges = [(0.2, 0.8), (math.inf, math.inf)]
    with patch("ier.infrequency.np.isnan", side_effect=AssertionError("integers have no NaN")):
        actual = infrequency(data, [0, 1], missing=policy, acceptable_ranges=ranges)
    np.testing.assert_array_equal(actual, [2, 2])
    unsigned_negative = infrequency(
        np.array([[0, 1]], dtype=np.uint8), [0, 1], acceptable_ranges=[(-5, -1), (-5, 1.5)]
    )
    np.testing.assert_array_equal(unsigned_negative, [1])


def test_fractional_ranges_round_inward_for_integer_categories() -> None:
    data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int16)
    np.testing.assert_array_equal(
        infrequency(data, [0, 1], acceptable_ranges=[(1.5, 3.5), (-math.inf, 2.999)]),
        [1, 0, 1, 2],
    )
    boolean = np.array([[False], [True]])
    np.testing.assert_array_equal(infrequency(boolean, [0], acceptable_ranges=[(0.5, 9)]), [1, 0])


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_single_precision_responses_compare_with_double_precision_bounds(dtype: type) -> None:
    # The stored response is slightly above 0.3. Rounding the bound to the
    # response dtype instead would wrongly accept it in the second range.
    data = np.array([[0.3, 0.3, 0.3]], dtype=dtype)
    stored = float(data[0, 0])
    assert stored > 0.3
    ranges = [(0.3, 0.4), (-math.inf, 0.3), (stored, stored)]
    np.testing.assert_array_equal(infrequency(data, [0, 1, 2], acceptable_ranges=ranges), [1.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_unrepresentable_integer_bounds_do_not_admit_float_neighbours(dtype: type) -> None:
    base = 2**70 + 1
    rounded = np.asarray(base, dtype=dtype)[()]
    assert int(rounded) != base
    data = np.array([[rounded] * 4, [np.inf] * 4, [-np.inf] * 4], dtype=dtype)
    ranges = [(base, base), (base, math.inf), (-math.inf, base), (-(10**400), 10**400)]
    # Python compares integers with integral or infinite floats exactly.
    exact_rows = [
        [int(value) if np.isfinite(value) else float(value) for value in row] for row in data
    ]
    reference = [
        sum(not lower <= value <= upper for value, (lower, upper) in zip(row, ranges, strict=True))
        for row in exact_rows
    ]
    assert reference == [2, 3, 3]
    np.testing.assert_array_equal(
        infrequency(data, [0, 1, 2, 3], acceptable_ranges=ranges), reference
    )


@pytest.mark.parametrize(("expected", "ranges"), [(None, None), ([1], [(1, 2)])])
def test_exactly_one_answer_specification_is_required(
    expected: list[float] | None, ranges: list[tuple[float, float]] | None
) -> None:
    for score in (infrequency, infrequency_flag):
        with pytest.raises(
            ValueError, match="exactly one of expected_responses or acceptable_ranges"
        ):
            score([[1]], [0], expected, acceptable_ranges=ranges)


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ([(1, 2), (3, 4)], "must have the same length"),
        ([(2, 1)], "low <= high"),
        ([(math.nan, 2)], "cannot be NaN"),
        ([(1, np.float32("nan"))], "cannot be NaN"),
        ([("1", 2)], "pairs of real numbers"),
        ([(True, 2)], "pairs of real numbers"),
        ([(1 + 0j, 2)], "pairs of real numbers"),
        ([(1, 2, 3)], "pairs of real numbers"),
        ([1], "pairs of real numbers"),
        (["12"], "pairs of real numbers"),
        ("12", "pairs of real numbers"),
        (5, "pairs of real numbers"),
        ([np.array(1)], "pairs of real numbers"),
    ],
)
def test_invalid_acceptable_ranges_raise_clear_errors(ranges: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        infrequency([[1]], [0], acceptable_ranges=ranges)  # type: ignore[arg-type]


def test_acceptable_ranges_accept_numeric_pair_containers() -> None:
    data = [[1, 5], [2, 4]]
    expected = infrequency(data, [0, 1], acceptable_ranges=[(1, 1), (5, 5)])
    np.testing.assert_array_equal(expected, [0, 2])
    for ranges in (
        np.array([[1, 1], [5, 5]]),
        [[1, 1], [5, 5]],
        ((np.int8(1), 1.0), (5, np.float32(5))),
    ):
        np.testing.assert_array_equal(infrequency(data, [0, 1], acceptable_ranges=ranges), expected)
