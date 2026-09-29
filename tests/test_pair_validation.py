"""Explicit item pairs must identify exactly the configured integer columns."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from ier import mad, semantic_ant, semantic_syn


@pytest.mark.parametrize("scorer", [mad, semantic_syn, semantic_ant])
@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [(0,)],
        [(0, 1, 2)],
        [(0, 1), (2,)],
        [None],
        [1],
        [(0.5, 1.5)],
        [(0, 1.0)],
        [(False, True)],
        [(0, np.bool_(True))],
        [("0", "1")],
        [(0, -1)],
        [(0, 3)],
    ],
)
def test_invalid_item_pairs_raise_value_error(scorer: Callable, pairs: Any) -> None:
    with pytest.raises(ValueError):
        scorer([[1, 2, 3], [3, 2, 1]], item_pairs=pairs)


@pytest.mark.parametrize("scorer", [mad, semantic_syn, semantic_ant])
def test_numpy_integer_indices_and_pair_order_are_preserved(scorer: Callable) -> None:
    data = [[1, 2, 3, 5], [5, 2, 3, 1], [3, 3, 3, 3]]
    expected = scorer(data, item_pairs=[(3, 0), (1, 2), (3, 0)])
    actual = scorer(data, item_pairs=[(np.int64(3), np.int32(0)), (1, 2), (3, 0)])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scorer", [semantic_syn, semantic_ant])
def test_semantic_pairs_require_distinct_items(scorer: Callable) -> None:
    with pytest.raises(ValueError, match="duplicate indices"):
        scorer([[1, 2, 3]], [(1, 1)])


def test_mad_keeps_its_existing_self_pair_policy() -> None:
    np.testing.assert_array_equal(
        mad([[5, 1, 3], [2, 4, 3]], item_pairs=[(0, 0)], scale_min=1, scale_max=5),
        [4, 2],
    )
