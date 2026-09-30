"""Computed infinite scores retain coverage and contextual reduction errors."""

from collections.abc import Callable

import numpy as np
import pytest

from ier import IndexOptions, composite, composite_flag, composite_summary
from ier.composite import _combine_scores

_MAX = float(np.finfo(float).max)
_MIN = float(np.nextafter(0.0, 1.0))
_DATA = np.array([[_MAX, _MAX], [1.0, 2.0], [-_MAX, -_MAX]])
_OPTIONS = IndexOptions(
    mad_positive_items=[0], mad_negative_items=[1], mad_scale_min=-_MAX, mad_scale_max=_MAX
)


@pytest.mark.parametrize("operation", [composite, composite_flag, composite_summary])
@pytest.mark.parametrize("method", ["sum", "max"])
@pytest.mark.parametrize("indices", [["mad"], ["mad", "longstring"]])
@pytest.mark.parametrize("workers", [1, 2])
def test_computed_infinite_scores_raise_contextual_range_error(
    operation: Callable[..., object], method: str, indices: list[str], workers: int
) -> None:
    data = _DATA.copy()
    data.flags.writeable = False
    with (
        np.errstate(all="raise"),
        pytest.raises(
            ValueError, match=rf"composite {method}.*finite float range.*index 0.*infinite score"
        ),
    ):
        operation(
            data,
            indices=indices,
            method=method,
            standardize=False,
            options=_OPTIONS,
            workers=workers,
        )
    np.testing.assert_array_equal(data, _DATA)


@pytest.mark.parametrize("method", ["sum", "max"])
@pytest.mark.parametrize("weight", [_MIN, 1.0, _MAX])
def test_weights_cannot_recover_a_computed_score_already_rounded_to_infinity(
    method: str, weight: float
) -> None:
    with (
        np.errstate(all="raise"),
        pytest.raises(ValueError, match=rf"composite {method}.*index 0.*infinite score"),
    ):
        composite(
            _DATA,
            indices=["mad"],
            method=method,
            standardize=False,
            options=_OPTIONS,
            weights={"mad": weight},
        )


@pytest.mark.parametrize("method", ["sum", "max"])
def test_ineligible_computed_infinite_scores_do_not_raise(method: str) -> None:
    mask = np.zeros(_DATA.shape, dtype=bool)
    mask[1] = True
    options = IndexOptions(
        mad_positive_items=[0],
        mad_negative_items=[1],
        mad_scale_min=-_MAX,
        mad_scale_max=_MAX,
        missing_applicable_mask=mask,
    )
    with np.errstate(all="raise"):
        result = composite_summary(
            _DATA,
            indices=["mad", "missing_rate"],
            method=method,
            standardize=False,
            options=options,
            min_valid_indices=2,
        )
    np.testing.assert_array_equal(result["composite"], [np.nan, 3.0, np.nan])
    np.testing.assert_array_equal(result["valid_index_counts"], [1, 2, 1])
    np.testing.assert_array_equal(result["indices"]["mad"], [np.inf, 3.0, np.inf])


@pytest.mark.parametrize("method", ["sum", "max"])
def test_failed_second_index_retains_single_component_coverage(method: str) -> None:
    with np.errstate(all="raise"):
        result = composite_summary(
            _DATA,
            indices=["mad", "semantic_syn"],
            method=method,
            standardize=False,
            options=_OPTIONS,
            min_valid_indices=2,
        )
    np.testing.assert_array_equal(result["composite"], [np.nan] * 3)
    np.testing.assert_array_equal(result["valid_index_counts"], [1] * 3)
    assert list(result["errors"]) == ["semantic_syn"]


@pytest.mark.parametrize("weight", [1.0, _MAX])
@pytest.mark.parametrize("strided", [False, True])
def test_maximum_discards_negative_infinity_and_retains_coverage(
    weight: float, strided: bool
) -> None:
    scores = {"irv": np.array([np.inf, np.inf, 2.0]), "longstring": np.array([0.5, np.nan, 0.5])}
    if strided:
        scores = {name: np.repeat(values, 2)[::2] for name, values in scores.items()}
    original = {name: values.copy() for name, values in scores.items()}
    for values in scores.values():
        values.flags.writeable = False
    counts = np.empty(3, dtype=int)
    with np.errstate(all="raise"):
        result = _combine_scores(
            scores,
            {},
            "max",
            False,
            {"irv": weight},
            min_valid_indices=2,
            valid_counts_out=counts,
            multipliers={"irv": -1.0},
        )
    np.testing.assert_array_equal(result, [0.5, np.nan, 0.5])
    np.testing.assert_array_equal(counts, [2, 1, 2])
    for name, values in scores.items():
        np.testing.assert_array_equal(values, original[name])


def test_unrepresentable_negative_maximum_reports_its_row() -> None:
    with np.errstate(all="raise"), pytest.raises(ValueError, match="composite max.*index 1"):
        _combine_scores(
            {"irv": np.array([np.inf, np.inf, 2.0]), "longstring": np.array([0.5, np.nan, 0.5])},
            {},
            "max",
            False,
            {"irv": _MAX},
            multipliers={"irv": -1.0},
        )


@pytest.mark.parametrize("weights", [None, {"mad": _MIN, "irv": _MAX}])
@pytest.mark.parametrize("minimum", [None, 2])
def test_infinite_mean_contributions_keep_ieee_scores_and_counts(
    weights: dict[str, float] | None, minimum: int | None
) -> None:
    counts = np.empty(5, dtype=int)
    with np.errstate(all="raise"):
        result = _combine_scores(
            {
                "mad": np.array([np.inf, 1.0, np.inf, np.inf, np.nan]),
                "irv": np.array([1.0, np.inf, np.inf, np.nan, 2.0]),
            },
            {},
            "mean",
            False,
            weights,
            min_valid_indices=minimum,
            valid_counts_out=counts,
            multipliers={"irv": -1.0},
        )
    expected = [np.inf, -np.inf, np.nan, np.inf, -2.0]
    if minimum is not None:
        expected[3:] = [np.nan, np.nan]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(counts, [2, 2, 2, 1, 1])


def test_batched_repair_preserves_infinite_rows_and_finite_selection_offsets() -> None:
    n_rows = 10003
    first = np.full(n_rows, 2.0)
    first[9000:9002] = np.inf
    second = np.full(n_rows, 0.5)
    second[9001] = np.nan
    counts = np.empty(n_rows, dtype=int)
    with np.errstate(all="raise"):
        result = _combine_scores(
            {"irv": first, "longstring": second},
            {},
            "max",
            False,
            {"irv": _MAX},
            min_valid_indices=2,
            valid_counts_out=counts,
            multipliers={"irv": -1.0},
        )
    expected = np.full(n_rows, 0.5)
    expected[9001] = np.nan
    expected_counts = np.full(n_rows, 2)
    expected_counts[9001] = 1
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(counts, expected_counts)


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
@pytest.mark.parametrize(
    ("indices", "minimum"),
    [
        (["mad"], None),
        (["mad"], 1),
        (["mad", "semantic_syn"], 2),
        (["mad", "longstring"], None),
        (["mad", "longstring"], 2),
    ],
)
def test_unavailable_calibration_has_consistent_counts_without_warnings(
    method: str, indices: list[str], minimum: int | None
) -> None:
    with np.errstate(all="raise"):
        result = composite_summary(
            _DATA,
            indices=indices,
            method=method,
            options=_OPTIONS,
            min_valid_indices=minimum,
        )
    expected_counts = [1] * 3 if "longstring" in indices else [0] * 3
    if "longstring" in indices and minimum is None:
        expected = [1 / np.sqrt(2), -np.sqrt(2), 1 / np.sqrt(2)]
    elif method == "sum" and minimum is None:
        expected = [0.0] * 3
    else:
        expected = [np.nan] * 3
    np.testing.assert_allclose(result["composite"], expected, rtol=1e-13, atol=0)
    np.testing.assert_array_equal(result["valid_index_counts"], expected_counts)


@pytest.mark.parametrize("method", ["sum", "max"])
@pytest.mark.parametrize("first_invalid", [0, 1])
def test_range_error_identifies_first_invalid_row_across_finite_and_infinite_inputs(
    method: str, first_invalid: int
) -> None:
    scores = {
        "longstring": np.array([2.0, np.nan])
        if first_invalid == 0
        else np.array([0.5, np.nan, 2.0]),
        "mad": np.array([np.nan, np.inf])
        if first_invalid == 0
        else np.array([np.nan, np.inf, np.nan]),
    }
    with (
        np.errstate(all="raise"),
        pytest.raises(ValueError, match=rf"composite {method}.*index {first_invalid}"),
    ):
        _combine_scores(scores, {}, method, False, {"longstring": _MAX})


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
def test_extreme_weights_repair_unavailable_calibration_with_consistent_coverage(
    method: str,
) -> None:
    with np.errstate(all="raise"):
        result = composite_summary(
            _DATA,
            indices=["mad", "longstring"],
            method=method,
            options=_OPTIONS,
            weights={"mad": _MIN, "longstring": _MAX},
            min_valid_indices=2,
        )
    np.testing.assert_array_equal(result["composite"], [np.nan] * 3)
    np.testing.assert_array_equal(result["valid_index_counts"], [1] * 3)


def test_mean_repair_with_only_infinite_rows_preserves_signed_results() -> None:
    counts = np.empty(3, dtype=int)
    with np.errstate(all="raise"):
        result = _combine_scores(
            {
                "mad": np.array([np.inf, -np.inf, np.inf]),
                "irv": np.array([-np.inf, np.inf, np.inf]),
            },
            {},
            "mean",
            False,
            {"mad": _MIN, "irv": _MAX},
            valid_counts_out=counts,
            multipliers={"irv": -1.0},
        )
    np.testing.assert_array_equal(result, [np.inf, -np.inf, np.nan])
    np.testing.assert_array_equal(counts, [2] * 3)


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
@pytest.mark.parametrize("strided", [False, True])
def test_unavailable_calibration_preserves_missing_readonly_inputs(
    method: str, strided: bool
) -> None:
    data = np.insert(_DATA, 2, [np.nan, np.nan], axis=0)
    if strided:
        data = np.repeat(data, 2, axis=1)[:, ::2]
    original = data.copy()
    data.flags.writeable = False
    with np.errstate(all="raise"):
        result = composite_summary(
            data,
            indices=["mad"],
            method=method,
            options=_OPTIONS,
            min_valid_indices=1,
        )
    np.testing.assert_array_equal(result["composite"], [np.nan] * 4)
    np.testing.assert_array_equal(result["valid_index_counts"], [0] * 4)
    np.testing.assert_array_equal(result["indices"]["mad"], [np.inf, 3.0, np.nan, np.inf])
    np.testing.assert_array_equal(data, original)
