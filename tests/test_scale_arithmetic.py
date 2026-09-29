"""Scale and paired-response arithmetic checked independently with Decimal."""

from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexOptions, acquiescence, mad, screen, semantic_ant, semantic_syn

_PAIRS = [(0, 5), (2, 4), (3, 1)]
_LEFT, _RIGHT = map(list, zip(*_PAIRS, strict=True))


def _decimal(value: float | int) -> Decimal:
    return (
        Decimal(int(value))
        if isinstance(value, (int, np.integer))
        else Decimal.from_float(float(value))
    )


def _data(lower: float, upper: float, layout: str) -> np.ndarray:
    fractions = np.array(
        [
            [0, 1, 0.25, 0.75, 0.5, 0.5],
            [0] * 6,
            [1] * 6,
            [0.5] * 6,
            [0, 0.25, 0.5, 0.75, 1, 0],
            [0] * 6,
        ]
    )
    data = lower * (1 - fractions) + upper * fractions
    data[4, 2:4] = np.nan
    data[5] = np.nan
    return data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)


def _acquiescence_reference(
    data: np.ndarray, lower: float, upper: float, paired: bool, ignore_nan: bool
) -> np.ndarray:
    result = []
    with localcontext() as context:
        context.prec = 800
        lo, width = _decimal(lower), _decimal(upper) - _decimal(lower)
        for row in data:
            if paired:
                values = [
                    value
                    for a, b in _PAIRS
                    for value in (row[a], row[b])
                    if not (np.isnan(row[a]) or np.isnan(row[b]))
                ]
                incomplete = len(values) != 2 * len(_PAIRS)
            else:
                values = row[~np.isnan(row)].tolist()
                incomplete = len(values) != len(row)
            if not values or (incomplete and not ignore_nan):
                result.append(np.nan)
            else:
                value = (sum(_decimal(value) for value in values) / len(values) - lo) / width
                result.append(float(min(Decimal(1), max(Decimal(0), value))))
    return np.asarray(result)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-1e308, 1e308),
        (5e307, 1.5e308),
        (-1.5e308, -5e307),
        (1.1, 1.1 + 16 * np.spacing(1.1)),
        (0, 4 * np.nextafter(0.0, 1.0)),
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_exceptional_acquiescence_matches_decimal(
    lower: float, upper: float, layout: str, paired: bool, ignore_nan: bool
) -> None:
    data = _data(lower, upper, layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _acquiescence_reference(data, lower, upper, paired, ignore_nan)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 13):
        actual = acquiescence(
            data,
            scale_min=lower,
            scale_max=upper,
            positive_items=_LEFT if paired else None,
            negative_items=_RIGHT if paired else None,
            na_rm=ignore_nan,
        )
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    ("lower", "dtype"), [(2**60, np.int64), (-(2**60), np.int64), (2**63, np.uint64)]
)
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("inferred", [False, True])
def test_acquiescence_preserves_adjacent_large_integers(
    lower: int, dtype: type, paired: bool, inferred: bool
) -> None:
    data = np.array(
        [[lower + value for value in row] for row in [[0, 8, 2, 6, 4, 4], [0] * 6, [8] * 6]],
        dtype=dtype,
    )
    expected = _acquiescence_reference(data, lower, lower + 8, paired, True)
    actual = acquiescence(
        data,
        scale_min=None if inferred else lower,
        scale_max=None if inferred else lower + 8,
        positive_items=_LEFT if paired else None,
        negative_items=_RIGHT if paired else None,
    )
    np.testing.assert_array_equal(actual, expected)


def test_narrow_explicit_bounds_preserve_cancellation_before_clipping() -> None:
    tiny = np.nextafter(0.0, 1.0)
    data = np.array(
        [[1e308, -1e308, 3 * tiny], [1e308, 1e308, 3 * tiny], [-1e308, -1e308, 3 * tiny]]
    )
    np.testing.assert_allclose(
        acquiescence(data, scale_min=0, scale_max=3 * tiny), [1 / 3, 1, 0], rtol=1e-15, atol=0
    )


def _pair_reference(
    data: np.ndarray, lower: float, upper: float, reflected: bool, semantic: bool, ignore_nan: bool
) -> np.ndarray:
    result = []
    with localcontext() as context:
        context.prec = 800
        reflection = _decimal(lower) + _decimal(upper)
        for row in data:
            differences = [
                abs(_decimal(row[a]) + _decimal(row[b]) - reflection)
                if reflected
                else abs(_decimal(row[a]) - _decimal(row[b]))
                for a, b in _PAIRS
                if not (np.isnan(row[a]) or np.isnan(row[b]))
            ]
            if not differences or (not ignore_nan and len(differences) != len(_PAIRS)):
                result.append(np.nan)
                continue
            difference = sum(differences) / len(differences)
            if not semantic:
                result.append(float(difference))
                continue
            observed = [_decimal(value) for value in row if not np.isnan(value)]
            mean = sum(observed) / len(observed)
            deviation = (sum((value - mean) ** 2 for value in observed) / len(observed)).sqrt()
            if not deviation:
                result.append(1.0 if difference <= Decimal.from_float(1e-8) else -1.0)
            else:
                result.append(float(min(Decimal(1), max(Decimal(-1), 1 - difference / deviation))))
    return np.asarray(result)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(-1e308, 1e308), (5e307, 1.5e308), (-1.5e308, -5e307), (1.1, 1.1 + 16 * np.spacing(1.1))],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("operation", ["mad", "mad_strict", "semantic_syn", "semantic_ant"])
def test_extreme_pair_scoring_matches_decimal(
    lower: float, upper: float, layout: str, operation: str
) -> None:
    data = _data(lower, upper, layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _pair_reference(
        data,
        lower,
        upper,
        operation != "semantic_syn",
        operation.startswith("semantic"),
        operation != "mad_strict",
    )
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 13):
        if operation.startswith("mad"):
            actual = mad(
                data,
                item_pairs=_PAIRS,
                scale_min=lower,
                scale_max=upper,
                na_rm=operation != "mad_strict",
            )
        elif operation == "semantic_ant":
            actual = semantic_ant(data, _PAIRS, scale_min=lower, scale_max=upper)
        else:
            actual = semantic_syn(data, _PAIRS)
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=0)
    np.testing.assert_array_equal(data, original)


def test_semantic_ratio_preserves_mean_differences_beyond_float_range() -> None:
    data = np.array([[1.5e308, -1.5e308], [1e308, -1e308]])
    np.testing.assert_array_equal(semantic_syn(data, [(0, 1)]), [-1, -1])
    # One overflowing difference and one zero difference have a finite average.
    data = np.array([[1e308, -1e308, 0, 0]])
    np.testing.assert_allclose(semantic_syn(data, [(0, 1), (2, 3)]), [1 - np.sqrt(2)], rtol=1e-14)
    # Even the mean difference can exceed float64 while its standardized ratio fits.
    data = np.array([[1.5e308, -1.5e308, 1.5e308, -1.5e308, 1.5e308, 1.5e308]])
    np.testing.assert_allclose(
        semantic_syn(data, [(0, 1), (2, 3), (4, 5)]), [1 - np.sqrt(2)], rtol=1e-14
    )


def test_wide_scale_acquiescence_remains_available_through_screening() -> None:
    result = screen(
        [[-1e308, -1e308], [0, 0], [1e308, 1e308], [np.nan, np.nan]],
        indices=["acquiescence"],
        options=IndexOptions(scale_min=-1e308, scale_max=1e308),
        thresholds={"acquiescence": 0.75},
        strict=True,
    )
    np.testing.assert_array_equal(result["scores"]["acquiescence"], [0, 0.5, 1, np.nan])
    np.testing.assert_array_equal(result["valid_index_counts"], [1, 1, 1, 0])
    np.testing.assert_array_equal(result["flag_counts"], [0, 0, 1, 0])


@pytest.mark.parametrize(
    ("lower", "dtype"), [(2**60, np.int64), (-(2**60), np.int64), (2**64 - 16, np.uint64)]
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("operation", ["mad", "semantic_syn", "semantic_ant"])
def test_large_integer_pair_scores_match_decimal(
    lower: int, dtype: type, layout: str, operation: str
) -> None:
    data = np.array(
        [
            [lower + value for value in row]
            for row in [[0, 8, 2, 6, 4, 4], [0] * 6, [8] * 6, [1, 2, 3, 4, 5, 6]]
        ],
        dtype=dtype,
    )
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _pair_reference(
        data, lower, lower + 8, operation != "semantic_syn", operation != "mad", True
    )
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 13):
        if operation == "mad":
            actual = mad(data, item_pairs=_PAIRS, scale_min=lower, scale_max=lower + 8)
        elif operation == "semantic_ant":
            actual = semantic_ant(data, _PAIRS, scale_min=lower, scale_max=lower + 8)
        else:
            actual = semantic_syn(data, _PAIRS)
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=5e-16)
    np.testing.assert_array_equal(data, original)


def test_large_integer_pair_means_preserve_fractional_bounds_and_cancellation() -> None:
    data = np.array([[2**60, -(2**60), 2**60 + 1, -(2**60) - 1, 1, 0]])
    pairs = [(0, 1), (2, 3), (4, 5)]
    np.testing.assert_allclose(
        mad(data, item_pairs=pairs, scale_min=-1.5, scale_max=2.75), [11 / 12], rtol=1e-15
    )


def test_large_integer_pair_mean_can_remain_unrepresentable() -> None:
    data = np.array([[2**60, 2**60 + 1]])
    np.testing.assert_array_equal(
        mad(data, item_pairs=[(0, 1)], scale_min=-1.7e308, scale_max=-1.6e308), [np.inf]
    )
    np.testing.assert_array_equal(
        semantic_ant(data, [(0, 1)], scale_min=-1.7e308, scale_max=-1.6e308), [-1]
    )
