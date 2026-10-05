"""Regression coverage for the shared bounded item-mean reduction."""

from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from ier._column_statistics import column_mean, column_mean_order
from ier.guttman import guttman
from ier.person_total import person_total


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16, np.uint32, np.int64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_column_means_match_numpy(ignore_nan: bool, dtype: type, layout: str) -> None:
    rng = np.random.default_rng(723)
    if np.issubdtype(dtype, np.integer):
        lower = 0 if np.issubdtype(dtype, np.unsignedinteger) else -8
        data = rng.integers(lower, 8, size=(157, 31), dtype=dtype)
    else:
        data = rng.normal(size=(157, 31)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        data[rng.random(data.shape) < 0.2] = np.nan
    data = data[::-2, ::2] if layout == "strided" else np.array(data, order=layout)
    data.flags.writeable = False
    expected = (
        np.nanmean(data, axis=0, dtype=float) if ignore_nan else np.mean(data, axis=0, dtype=float)
    )
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual = column_mean(data, ignore_nan=ignore_nan)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_column_means_leave_empty_items_unavailable_without_warning() -> None:
    data = np.array([[np.nan, 1.0, np.nan], [np.nan, 3.0, 5.0]])
    np.testing.assert_array_equal(column_mean(data, ignore_nan=True), [np.nan, 2.0, 5.0])
    np.testing.assert_array_equal(column_mean(data, ignore_nan=False), [np.nan, 2.0, np.nan])


def test_column_mean_masks_are_bounded() -> None:
    data = np.zeros((513, 31))
    data[3, 4] = np.nan
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100),
        patch("ier._column_statistics.np.isnan", wraps=np.isnan) as masks,
    ):
        column_mean(data, ignore_nan=True)
    assert masks.call_count > 1
    assert all(call.args[0].size <= 100 for call in masks.call_args_list)


def _decimal_mean(values: np.ndarray, *, ignore_nan: bool) -> complex | float:
    observed = values[~np.isnan(values)] if ignore_nan else values
    if not len(observed) or np.isnan(observed).any():
        return np.nan
    with localcontext() as context:
        context.prec = 800
        real = float(
            sum(Decimal.from_float(float(value.real)) for value in observed) / len(observed)
        )
        if np.iscomplexobj(values):
            imag = float(
                sum(Decimal.from_float(float(value.imag)) for value in observed) / len(observed)
            )
            return complex(real, imag)
        return real


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
def test_extreme_column_means_match_decimal(
    ignore_nan: bool, layout: str, missing: bool, complex_values: bool
) -> None:
    largest = np.finfo(float).max
    smallest = np.nextafter(0.0, 1.0)
    data = np.array(
        [
            [largest] * 6,
            [-largest] * 6,
            [largest, largest, -largest, -largest, largest / 2, largest / 2],
            [1e308, 1e308, 1e308, 5e307, 5e307, 5e307],
            [smallest, 2 * smallest, 3 * smallest, 4 * smallest, 5 * smallest, 6 * smallest],
            [0, 0, 0, 0, 0, 0],
        ]
    ).T
    if complex_values:
        complex_data = np.empty(data.shape, dtype=complex)
        complex_data.real = data
        complex_data.imag = data[:, ::-1]
        data = complex_data
    if missing:
        data[1::2, 1::2] = np.nan
        data[:, -1] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = [_decimal_mean(column, ignore_nan=ignore_nan) for column in data.T]
    # Force both within-block and between-block overflow cases.
    for budget in [12, 100]:
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget):
            actual = column_mean(data, ignore_nan=ignore_nan)
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_single_precision_item_means_accumulate_in_double_precision(ignore_nan: bool) -> None:
    data = np.array([[1e8, 1e8 + 8], [1e8 + 8, 1e8 + 24]], dtype=np.float32)
    np.testing.assert_array_equal(column_mean(data, ignore_nan=ignore_nan), [100000004, 100000016])
    tied = np.array([[1e8, 1e8], [1e8 + 8, 1e8]], dtype=np.float32)
    np.testing.assert_array_equal(
        guttman(tied, na_rm=ignore_nan), guttman(tied.astype(float), na_rm=ignore_nan)
    )


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_infinite_column_means_keep_numpy_availability(ignore_nan: bool) -> None:
    data = np.array([[1, 1, np.inf, np.inf], [np.inf, -np.inf, -np.inf, np.nan]])
    np.testing.assert_array_equal(
        column_mean(data, ignore_nan=ignore_nan),
        [np.inf, -np.inf, np.nan, np.inf if ignore_nan else np.nan],
    )


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_item_means_preserve_shape(ignore_nan: bool, shape: tuple[int, int]) -> None:
    np.testing.assert_array_equal(
        column_mean(np.empty(shape), ignore_nan=ignore_nan), np.full(shape[1], np.nan)
    )


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_public_scores_preserve_overflowing_item_means(ignore_nan: bool) -> None:
    data = np.array([[1, 4, 7], [4, 1, 6], [8, 2, 5]], dtype=float)
    for scorer in (person_total, guttman):
        np.testing.assert_allclose(
            scorer(data * 1e307, na_rm=ignore_nan),
            scorer(data, na_rm=ignore_nan),
            rtol=1e-13,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    ("ignore_nan", "expected"), [(True, [2, 3, 4, 0, 1]), (False, [2, 4, 0, 1, 3])]
)
def test_item_order_is_easiest_first_with_column_ties_and_missing_means_last(
    ignore_nan: bool, expected: list[int]
) -> None:
    data = np.array([[1.0, np.nan, 3.0, 2.0, 3.0], [2.0, np.nan, 1.0, np.nan, 1.0]])
    np.testing.assert_array_equal(column_mean_order(data, ignore_nan=ignore_nan), expected)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64])
def test_integer_item_order_breaks_exact_total_ties_by_column(dtype: type) -> None:
    data = np.array([[0, 2, 1, 1, 3], [1, 1, 2, 1, 0]], dtype=dtype)
    np.testing.assert_array_equal(column_mean_order(data, ignore_nan=True), [1, 2, 4, 3, 0])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_integer_item_means_and_profiles_preserve_exact_totals(
    dtype: type, layout: str, ignore_nan: bool
) -> None:
    data = np.random.default_rng(825).bit_generator.random_raw((19, 7)).view(dtype)
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    totals = [sum(int(value) for value in column) for column in data.T]
    expected = [total / len(data) for total in totals]
    expected_profile = [(total - min(totals)) / len(data) for total in totals]
    expected_order = sorted(range(data.shape[1]), key=lambda column: -totals[column])
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 13):
        actual = column_mean(data, ignore_nan=ignore_nan)
        profile = column_mean(data, ignore_nan=ignore_nan, center_integers=True)
        order = column_mean_order(data, ignore_nan=ignore_nan)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(profile, expected_profile)
    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_integer_item_means_preserve_cancellation_across_blocks(ignore_nan: bool) -> None:
    data = np.array([[2**63 - 1, 2**60], [-(2**63), 2**60 + 1], [2, 2**60 + 2]])
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 2):
        np.testing.assert_array_equal(
            column_mean(data, ignore_nan=ignore_nan), [1 / 3, float(2**60 + 1)]
        )


@pytest.mark.parametrize("ignore_nan", [False, True])
@pytest.mark.parametrize(
    ("dtype", "offset"), [(np.int64, 2**60), (np.int64, -(2**60)), (np.uint64, 2**64 - 32)]
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_person_total_preserves_large_integer_profiles(
    ignore_nan: bool, dtype: type, offset: int, layout: str
) -> None:
    base = np.random.default_rng(521).integers(0, 16, size=(31, 7))
    base[0] = 3
    data = base.astype(dtype) + offset
    if layout == "strided":
        data, base = data[::-1, ::-1], base[::-1, ::-1]
    else:
        data = np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 25),
        patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", 25),
    ):
        actual = person_total(data, na_rm=ignore_nan)
    np.testing.assert_allclose(actual, person_total(base, na_rm=ignore_nan), rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(data, original)


def test_strict_person_total_missing_scan_stays_bounded() -> None:
    data = np.arange(100, dtype=float).reshape(20, 5)
    data[-1, -1] = np.nan
    with (
        patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", 11),
        patch("ier.person_total.np.isnan", wraps=np.isnan) as scans,
    ):
        actual = person_total(data, na_rm=False)
    assert np.isnan(actual).all()
    assert max(call.args[0].size for call in scans.call_args_list) <= 11


@pytest.mark.parametrize(
    ("dtype", "offset"), [(np.int64, 2**60), (np.int64, -(2**60)), (np.uint64, 2**64 - 1_000_000)]
)
@pytest.mark.parametrize(("categories", "step"), [(7, 1), (19, 1000), (121, 1000)])
def test_guttman_preserves_large_integer_categories(
    dtype: type, offset: int, categories: int, step: int
) -> None:
    base = np.random.default_rng(25).integers(0, categories, size=(91, 13)) * step
    data = base.astype(dtype) + offset
    original = data.copy()
    data.flags.writeable = False
    for normalize in (False, True):
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 31):
            actual = guttman(data, normalize=normalize)
        np.testing.assert_array_equal(actual, guttman(base, normalize=normalize))
    np.testing.assert_array_equal(data, original)


def test_guttman_ranks_nearby_large_items_even_with_a_distant_item() -> None:
    lower = 2**60
    data = np.array([[0, lower + 1, lower + 4], [0, lower + 3, lower + 1]])
    totals = [sum(int(value) for value in column) for column in data.T]
    order = sorted(range(3), key=lambda column: -totals[column])
    expected = [
        sum(row[order[a]] < row[order[b]] for a in range(3) for b in range(a + 1, 3))
        for row in data
    ]
    np.testing.assert_array_equal(guttman(data, normalize=False), expected)
