"""Deliberate consumer mistakes; the typing gate requires these diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ier import (
    MatrixLike,
    composite,
    composite_flag,
    composite_probability,
    composite_scores,
    evenodd,
)

if TYPE_CHECKING:
    import numpy as np


def incorrect_return_shapes(responses: MatrixLike, diagnostics: bool) -> None:
    _scores_as_tuple: tuple[np.ndarray, dict[str, str]] = composite(
        responses
    )  # expect: invalid-assignment
    _diagnostics_as_array: np.ndarray = composite(
        responses, return_diagnostics=True
    )  # expect: invalid-assignment
    _flags_as_array: np.ndarray = composite_flag(responses)  # expect: invalid-assignment
    _missing_diagnostics: tuple[np.ndarray, np.ndarray, dict[str, str]] = composite_flag(
        responses
    )  # expect: invalid-assignment
    _probability_as_array: np.ndarray = composite_probability(
        responses, return_diagnostics=True
    )  # expect: invalid-assignment
    _dynamic_as_array: np.ndarray = composite_probability(
        responses, return_diagnostics=diagnostics
    )  # expect: invalid-assignment
    composite(responses, method="average")  # expect: invalid-argument-type
    composite_scores({"irv": [0.1]}, unsupported="skip")  # expect: invalid-argument-type


def incorrect_consistency_calls(responses: MatrixLike) -> None:
    _diagnostics_as_array: np.ndarray = evenodd(
        responses, [4, 4], diag=True
    )  # expect: invalid-assignment
    evenodd(responses, [4, 4], method="pairs")  # expect: invalid-argument-type


def incorrect_repetition_and_effort_calls(responses: MatrixLike) -> None:
    from ier import autocorrelation, response_time_effort

    _lags_as_array: np.ndarray = autocorrelation(
        responses, return_lags=True
    )  # expect: invalid-assignment
    autocorrelation(responses, statistic="maximum")  # expect: no-matching-overload
    _rapid_as_array: np.ndarray = response_time_effort(
        responses, return_item_flags=True
    )  # expect: invalid-assignment


def incorrect_person_fit_calls(responses: MatrixLike) -> None:
    from ier import IndexOptions, ht_flag

    _flags_as_array: np.ndarray = ht_flag(responses)  # expect: invalid-assignment
    IndexOptions(person_fit_ncat="5")  # expect: invalid-argument-type
