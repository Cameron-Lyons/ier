"""Regression coverage for bounded missing-response and attention-check scoring."""

from unittest.mock import patch

import numpy as np
import pytest

import ier._row_statistics as row_statistics
from ier import infrequency, missing_rate
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
