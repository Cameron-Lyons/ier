"""Deliberate consumer mistakes; the typing gate requires these diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ier import MatrixLike, composite, composite_flag, composite_probability

if TYPE_CHECKING:
    import numpy as np


def incorrect_return_shapes(responses: MatrixLike, diagnostics: bool) -> None:
    _scores_as_tuple: tuple[np.ndarray, dict[str, str]] = composite(responses)  # expect: assignment
    _diagnostics_as_array: np.ndarray = composite(
        responses, return_diagnostics=True
    )  # expect: assignment
    _flags_as_array: np.ndarray = composite_flag(responses)  # expect: assignment
    _missing_diagnostics: tuple[np.ndarray, np.ndarray, dict[str, str]] = composite_flag(
        responses
    )  # expect: assignment
    _probability_as_array: np.ndarray = composite_probability(
        responses, return_diagnostics=True
    )  # expect: assignment
    _dynamic_as_array: np.ndarray = composite_probability(
        responses, return_diagnostics=diagnostics
    )  # expect: assignment
    composite(responses, method="average")  # expect: call-overload
