"""Return-shape controls reject values that only look Boolean."""

from collections.abc import Callable

import numpy as np
import pytest

from ier import (
    autocorrelation,
    composite,
    composite_flag,
    composite_probability,
    evenodd,
    longstring,
    longstring_scores,
    mahad,
    psychant,
    psychsyn,
    response_time_effort,
)

_RETURN_SHAPE_CALLS: list[tuple[str, str, Callable[[object], object]]] = [
    ("autocorrelation", "return_lags", lambda value: autocorrelation(object(), return_lags=value)),
    (
        "composite",
        "return_diagnostics",
        lambda value: composite(object(), return_diagnostics=value),
    ),
    (
        "composite_flag",
        "return_diagnostics",
        lambda value: composite_flag(object(), return_diagnostics=value),
    ),
    (
        "composite_probability",
        "return_diagnostics",
        lambda value: composite_probability(object(), return_diagnostics=value),
    ),
    ("evenodd", "diag", lambda value: evenodd(object(), [4], diag=value)),
    ("longstring", "avg", lambda value: longstring("aab", avg=value)),
    ("longstring_scores", "avg", lambda value: longstring_scores(object(), avg=value)),
    ("mahad", "flag", lambda value: mahad(object(), flag=value)),
    ("psychant", "diag", lambda value: psychant(object(), diag=value)),
    ("psychsyn", "diag", lambda value: psychsyn(object(), diag=value)),
    (
        "psychsyn item info",
        "_return_item_info",
        lambda value: psychsyn(object(), _return_item_info=value),
    ),
    (
        "response_time_effort",
        "return_item_flags",
        lambda value: response_time_effort(object(), return_item_flags=value),
    ),
]


@pytest.mark.parametrize(("function_name", "option_name", "call"), _RETURN_SHAPE_CALLS)
@pytest.mark.parametrize(
    "value",
    [1, 0, np.bool_(True), "true", np.array(False), [True], None],
)
def test_return_shape_options_require_python_booleans_before_scoring(
    function_name: str,
    option_name: str,
    call: Callable[[object], object],
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"^{option_name} must be a boolean$"):
        call(value)


def test_python_booleans_keep_declared_return_shapes() -> None:
    responses = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [1.0, 3.0, 2.0, 4.0],
        ]
    )

    assert isinstance(autocorrelation(responses, max_lag=1, return_lags=False), np.ndarray)
    assert isinstance(autocorrelation(responses, max_lag=1, return_lags=True), tuple)
    assert isinstance(evenodd(responses, [4], diag=False), np.ndarray)
    assert isinstance(evenodd(responses, [4], diag=True), tuple)
    assert isinstance(mahad(responses, flag=False), np.ndarray)
    assert isinstance(mahad(responses, flag=True), tuple)
    assert isinstance(response_time_effort(responses, return_item_flags=False), np.ndarray)
    assert isinstance(response_time_effort(responses, return_item_flags=True), tuple)
    assert isinstance(longstring("aab", avg=False), tuple)
    assert isinstance(longstring("aab", avg=True), float)
