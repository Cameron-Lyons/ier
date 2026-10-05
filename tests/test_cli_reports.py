"""Command-line reports validate respondent alignment once for every serializer."""

from __future__ import annotations

import numpy as np
import pytest

from ier._cli_composite import CompositeReport, ResponseTimeReport


@pytest.mark.parametrize("respondent_ids", [["a", "b"], ["a", "b", "c", "d"], []])
def test_reports_reject_respondent_ids_of_another_length(respondent_ids: list[str]) -> None:
    scores = np.array([0.1, 0.2, 0.3])
    flags = np.array([False, False, True])

    with pytest.raises(ValueError, match="respondent ID count must match result length"):
        CompositeReport(scores, "mean", respondent_ids=respondent_ids)
    with pytest.raises(ValueError, match="respondent ID count must match result length"):
        ResponseTimeReport(scores, flags, "median", "low", 0.3, respondent_ids)


def test_reports_accept_aligned_or_absent_respondent_ids() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    flags = np.array([False, False, True])
    ids = ["a", "b", "c"]

    assert CompositeReport(scores, "mean", respondent_ids=ids).respondent_ids == ids
    assert CompositeReport(scores, "mean").respondent_ids is None
    assert ResponseTimeReport(scores, flags, "median", "low", 0.3, ids).respondent_ids == ids
    assert ResponseTimeReport(scores, flags, "median", "low", 0.3).respondent_ids is None
