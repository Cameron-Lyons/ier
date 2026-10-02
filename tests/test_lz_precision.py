"""High-precision proofs of calibrated lz likelihood centering and variance."""

from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from ier import lz_flag


def _decimal_scores(
    responses: np.ndarray,
    difficulties: np.ndarray,
    discriminations: np.ndarray,
    abilities: np.ndarray,
    *,
    na_rm: bool,
) -> np.ndarray:
    """Evaluate Bernoulli moments from an independently calibrated IRT model."""
    with localcontext() as context:
        context.prec = 450
        minimum = Decimal.from_float(1e-10)
        maximum = Decimal.from_float(1 - 1e-10)
        scores = []
        for row, ability in zip(responses, abilities, strict=True):
            if (np.isnan(row).any() and not na_rm) or np.isnan(ability):
                scores.append(np.nan)
                continue
            observed = False
            centered_likelihood = Decimal(0)
            variance = Decimal(0)
            for response, difficulty, discrimination in zip(
                row, difficulties, discriminations, strict=True
            ):
                if np.isnan(response):
                    continue
                observed = True
                if np.isnan(difficulty) or np.isnan(discrimination):
                    centered_likelihood = Decimal("NaN")
                    break
                predictor = Decimal.from_float(float(discrimination)) * (
                    Decimal.from_float(float(ability)) - Decimal.from_float(float(difficulty))
                )
                probability = 1 / (1 + (-predictor).exp())
                clipped = min(max(probability, minimum), maximum)
                log_odds = predictor if probability == clipped else (clipped / (1 - clipped)).ln()
                centered_likelihood += (Decimal.from_float(float(response)) - clipped) * log_odds
                variance += clipped * (1 - clipped) * log_odds * log_odds
            if not observed or centered_likelihood.is_nan():
                scores.append(np.nan)
            elif variance == 0:
                scores.append(0.0)
            else:
                scores.append(float(centered_likelihood / variance.sqrt()))
    return np.asarray(scores)


@pytest.mark.parametrize("scale", [1e-4, 1e-8, 1e-12, 1e-15, 1e-100, 1e-200, 1e-300])
@pytest.mark.parametrize("model", ["1pl", "2pl"])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_neutral_item_likelihoods_retain_calibrated_scores(
    scale: float, model: str, layout: str, na_rm: bool
) -> None:
    data = np.asarray([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [1, np.nan, 0, 0], [np.nan] * 4])
    if layout == "strided":
        backing = np.zeros((2 * len(data), 2 * data.shape[1]))
        backing[::2, ::2] = data
        data = backing[::2, ::2]
    else:
        data = np.array(data, order=layout)
    data.flags.writeable = False
    original = data.copy()
    difficulties = np.asarray([-1, -2, 1, 2], dtype=float) * scale
    discriminations = np.asarray([0.5, 1.5, 0.5, 1.5]) if model == "2pl" else np.ones(4)
    abilities = np.zeros(len(data))
    expected = _decimal_scores(data, difficulties, discriminations, abilities, na_rm=na_rm)
    with np.errstate(all="raise"), patch("ier.lz._LZ_BATCH_ELEMENTS", 7):
        scores, flags = lz_flag(
            data,
            difficulty=difficulties,
            discrimination=discriminations,
            theta=abilities,
            model=model,
            threshold=-1.5,
            na_rm=na_rm,
        )
    np.testing.assert_allclose(scores, expected, rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(flags, expected < -1.5)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("scale", [1e-300, 1e-12, 0.1, 1.0, 10.0, 1000.0])
@pytest.mark.parametrize("na_rm", [False, True])
def test_likelihood_clipping_and_missing_items_match_decimal(scale: float, na_rm: bool) -> None:
    data = np.asarray(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [1, np.nan, 1, np.nan], [np.nan] * 4]
    )
    difficulties = np.asarray([-1.3, 0.8, -0.2, 1.7]) * scale
    discriminations = np.asarray([0.8, 1.1, 1.9, 0.3])
    abilities = np.asarray([0.1, -0.4, 0.8, 0.1, -0.1]) * scale
    expected = _decimal_scores(data, difficulties, discriminations, abilities, na_rm=na_rm)
    scores, flags = lz_flag(
        data,
        difficulty=difficulties,
        discrimination=discriminations,
        theta=abilities,
        threshold=-1.5,
        na_rm=na_rm,
    )
    np.testing.assert_allclose(scores, expected, rtol=3e-14, atol=1e-14)
    np.testing.assert_array_equal(flags, expected < -1.5)


def test_missing_parameter_does_not_change_tiny_observed_likelihoods() -> None:
    data = np.asarray([[1, 0, np.nan], [0, 1, 1], [np.nan, np.nan, np.nan]])
    difficulties = np.asarray([-1e-300, 2e-300, np.nan])
    discriminations = np.asarray([1.0, 1.0, np.nan])
    abilities = np.zeros(3)
    expected = _decimal_scores(data, difficulties, discriminations, abilities, na_rm=True)
    with np.errstate(all="raise"):
        scores, flags = lz_flag(
            data, difficulty=difficulties, discrimination=discriminations, theta=abilities
        )
    np.testing.assert_allclose(scores, expected, rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(flags, expected < -1.96)
