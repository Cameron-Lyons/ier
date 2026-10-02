"""
Standardized Log-Likelihood (lz) person-fit statistic for detecting aberrant response patterns.

The lz statistic is an IRT-based person-fit index that measures the discrepancy between
observed and expected response patterns. Under proper conditions, lz approximately follows
a standard normal distribution, with negative values indicating responses that are
inconsistent with the expected pattern. The approximation is strongest when item and person
parameters come from an appropriate, independently fitted IRT model.

References:
- Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness measurement with
  polychotomous item response models and standardized indices. British Journal of
  Mathematical and Statistical Psychology, 38(1), 67-86.
- Meijer, R. R., & Sijtsma, K. (2001). Methodology review: Evaluating person fit.
  Applied Psychological Measurement, 25(2), 107-135.
"""

import math
from fractions import Fraction

import numpy as np

from ier._column_statistics import column_mean
from ier._row_statistics import row_slices, row_sum
from ier._statistics import logistic_transform
from ier._validation import MatrixLike, validate_matrix_input, validate_score_array

_LZ_BATCH_ELEMENTS = 10_240
_MIN_PROBABILITY = 1e-10
_MAX_PROBABILITY = 1 - _MIN_PROBABILITY
_MIN_LOG_ODDS = math.log(_MIN_PROBABILITY / (1 - _MIN_PROBABILITY))
_MAX_LOG_ODDS = math.log(_MAX_PROBABILITY / (1 - _MAX_PROBABILITY))


def lz(
    x: MatrixLike,
    difficulty: np.ndarray | list[float] | None = None,
    discrimination: np.ndarray | list[float] | None = None,
    theta: np.ndarray | list[float] | None = None,
    model: str = "2pl",
    na_rm: bool = True,
) -> np.ndarray:
    """
    Calculate standardized log-likelihood (lz) person-fit statistic.

    The lz statistic measures how well each person's response pattern fits the
    expected pattern under an Item Response Theory (IRT) model. Negative values
    indicate aberrant or unexpected response patterns, potentially suggesting
    careless or random responding.

    Interpret scores cautiously when parameters are estimated from the same
    response matrix passed to this function. Those estimates are convenient
    fallbacks, not replacements for a calibrated IRT model, and can make the
    usual lz distributional interpretation less reliable. Polytomous responses
    are dichotomized at the observed midpoint before scoring, which discards
    category information and may be inappropriate for ordered-rating scales
    unless that simplification is acceptable for the analysis.

    Parameters:
    - x: A matrix of dichotomous data (0/1) where rows are individuals and
         columns are items. Polytomous data is dichotomized at the observed midpoint.
    - difficulty: One-dimensional real numeric vector of item difficulties (b).
                  If None, estimated from data. NaN marks unavailable parameters.
    - discrimination: One-dimensional real numeric vector of item discriminations (a).
                     If None and model="2pl", estimated from data. Ignored if model="1pl".
                     NaN marks unavailable parameters.
    - theta: One-dimensional real numeric vector of person ability estimates.
             If None, estimated from data. NaN marks unavailable estimates.
    - model: IRT model to use. "1pl" (Rasch) or "2pl" (default).
    - na_rm: Boolean indicating whether to handle missing values.

    Returns:
    - A numpy array of lz values for each individual. Values significantly below
      -1.96 may indicate careless responding (at alpha=0.05) when the IRT model
      and parameter estimates support that approximation.

    Raises:
    - ValueError: If inputs are invalid.

    Example:
        >>> data = [[1, 1, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        >>> lz_scores = lz(data)
        >>> print(lz_scores)
    """
    x_array = validate_matrix_input(x, check_type=False)

    if model not in ["1pl", "2pl"]:
        raise ValueError("model must be '1pl' or '2pl'")

    x_binary = _dichotomize(x_array)

    if difficulty is not None:
        b = validate_score_array(difficulty, name="difficulty")
        if len(b) != x_array.shape[1]:
            raise ValueError("difficulty length must match number of items")
    else:
        b = _estimate_difficulty(x_binary, na_rm=na_rm)

    if model == "2pl":
        if discrimination is not None:
            a = validate_score_array(discrimination, name="discrimination")
            if len(a) != x_array.shape[1]:
                raise ValueError("discrimination length must match number of items")
        else:
            a = _estimate_discrimination(x_binary, na_rm=na_rm)
    else:
        a = np.ones(x_array.shape[1])

    if theta is not None:
        theta_arr = validate_score_array(theta, name="theta")
        if len(theta_arr) != x_array.shape[0]:
            raise ValueError("theta length must match number of respondents")
    else:
        theta_arr = _estimate_theta(x_binary, a, b, na_rm=na_rm)

    lz_values = _compute_lz(x_binary, a, b, theta_arr, na_rm=na_rm)

    return lz_values


def lz_flag(
    x: MatrixLike,
    difficulty: np.ndarray | list[float] | None = None,
    discrimination: np.ndarray | list[float] | None = None,
    theta: np.ndarray | list[float] | None = None,
    model: str = "2pl",
    threshold: float = -1.96,
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate lz scores and flag potential careless responders.

    Parameters:
    - x: A matrix of dichotomous data (0/1).
    - difficulty: Array of item difficulty parameters.
    - discrimination: Array of item discrimination parameters.
    - theta: Array of person ability estimates.
    - model: IRT model ("1pl" or "2pl").
    - threshold: lz threshold below which to flag (default -1.96 for alpha=0.05).
    - na_rm: Handle missing values.

    Returns:
    - Tuple of (lz_scores, flags) where flags is True for suspected careless responders.

    Example:
        >>> data = [[1, 1, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        >>> scores, flags = lz_flag(data)
        >>> print(flags)
        [False, False, True]
    """
    scores = lz(
        x,
        difficulty=difficulty,
        discrimination=discrimination,
        theta=theta,
        model=model,
        na_rm=na_rm,
    )

    flags = np.zeros(len(scores), dtype=bool)
    valid_mask = ~np.isnan(scores)
    flags[valid_mask] = scores[valid_mask] < threshold

    return scores, flags


def _dichotomize(x: np.ndarray) -> np.ndarray:
    """Reuse binary responses or dichotomize at the observed midpoint in batches."""
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        if not np.all((block == 0) | (block == 1) | np.isnan(block)):
            break
    else:
        return x

    lower, upper = np.nanmin(x), np.nanmax(x)
    comparison: np.ufunc = np.greater
    if x.dtype.kind in "iu":
        # An integer is above the exact midpoint iff it is above its floor.
        # Python arithmetic preserves labels near either 64-bit limit.
        midpoint = np.asarray([(int(lower) + int(upper)) // 2], dtype=x.dtype)
    else:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            center = (lower + upper) / 2
            if np.isinf(center) and np.isfinite(lower) and np.isfinite(upper):
                center = lower / 2 + upper / 2
            midpoint = np.asarray([center], dtype=x.dtype)
        center = midpoint[0]
        if np.isfinite(center):
            # A midpoint rounded upward can equal an observed category that is
            # strictly above the true midpoint. Include that category as well.
            exact_center = (
                Fraction(*lower.as_integer_ratio()) + Fraction(*upper.as_integer_ratio())
            ) / 2
            if Fraction(*center.as_integer_ratio()) > exact_center:
                comparison = np.greater_equal
    result = np.empty(x.shape)
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        comparison(block, midpoint, out=result[start:stop])
        np.copyto(result[start:stop], np.nan, where=np.isnan(block))
    return result


def _estimate_difficulty(x: np.ndarray, na_rm: bool = True) -> np.ndarray:
    """Estimate item difficulty from proportion correct."""
    p = column_mean(x, ignore_nan=na_rm)
    if np.issubdtype(x.dtype, np.floating):
        p = p.astype(x.dtype, copy=False)
    p = np.clip(p, 0.001, 0.999)
    b: np.ndarray = -np.log(p / (1 - p))
    return b


def _estimate_discrimination(x: np.ndarray, na_rm: bool = True) -> np.ndarray:
    """Estimate binary-item discrimination with bounded point-biserial reductions."""
    n_items = x.shape[1]
    a = np.ones(n_items)
    total_score = row_sum(x, ignore_nan=na_rm)

    # A missing total invalidates every correlation under the strict policy.
    if not np.isfinite(total_score).all() or np.min(total_score) == np.max(total_score):
        return a

    counts = np.zeros(n_items, dtype=np.intp)
    response_sums = np.zeros(n_items)
    score_sums = np.zeros(n_items)
    for start, stop in row_slices(len(x), n_items):
        block = x[start:stop]
        valid = ~np.isnan(block)
        counts += np.sum(valid, axis=0, dtype=np.intp)
        response_sums += np.sum(block, axis=0, dtype=float, where=valid)
        score_sums += np.einsum("ij,i->j", valid, total_score[start:stop])

    observed = counts > 0
    proportions = np.divide(response_sums, counts, out=np.zeros(n_items), where=observed)
    score_means = np.divide(score_sums, counts, out=np.zeros(n_items), where=observed)

    covariance = np.zeros(n_items)
    score_variance = np.zeros(n_items)
    # Budget both centered matrices; each item's totals use its observed respondents.
    for start, stop in row_slices(len(x), 2 * n_items):
        block = x[start:stop]
        valid = ~np.isnan(block)
        centered_responses = np.zeros(block.shape)
        centered_scores = np.zeros(block.shape)
        np.subtract(block, proportions, out=centered_responses, where=valid)
        np.subtract(total_score[start:stop, None], score_means, out=centered_scores, where=valid)
        covariance += np.einsum("ij,ij->j", centered_responses, centered_scores)
        score_variance += np.einsum("ij,ij->j", centered_scores, centered_scores)
        del centered_responses, centered_scores

    # Binary responses have centered sum of squares n * p * (1 - p).
    denominator = np.sqrt(score_variance * counts * proportions * (1.0 - proportions))
    usable = denominator > 0
    correlations = np.divide(covariance, denominator, out=np.zeros(n_items), where=usable)
    np.clip(correlations, -0.99, 0.99, out=correlations)
    a[usable] = np.clip(
        correlations[usable] * 1.7 / np.sqrt(1.0 - correlations[usable] ** 2), 0.2, 3.0
    )
    return a


def _estimate_theta(x: np.ndarray, a: np.ndarray, b: np.ndarray, na_rm: bool = True) -> np.ndarray:
    """Estimate abilities in bounded batches, omitting missing items when requested."""
    theta = np.empty(x.shape[0])
    batch_rows = max(1, _LZ_BATCH_ELEMENTS // x.shape[1])
    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        theta[start:stop] = _ml_theta_batch(x[start:stop], a, b, na_rm=na_rm)
    return theta


def _ml_theta_batch(
    responses: np.ndarray, a: np.ndarray, b: np.ndarray, *, na_rm: bool = True
) -> np.ndarray:
    """Apply safeguarded Newton iterations to one response batch."""
    theta = np.full(len(responses), np.nan)
    missing = np.isnan(responses)
    observed = ~missing if na_rm and np.any(missing) else None
    if observed is None:
        available = ~np.any(missing, axis=1)
        all_correct = np.all(responses == 1, axis=1)
        all_incorrect = np.all(responses == 0, axis=1)
    else:
        available = np.any(observed, axis=1)
        all_correct = available & np.all((responses == 1) | missing, axis=1)
        all_incorrect = available & np.all((responses == 0) | missing, axis=1)
    theta[all_correct] = 3.0
    theta[all_incorrect] = -3.0

    interior = available & ~(all_correct | all_incorrect)
    active_responses = responses[interior]
    if len(active_responses) == 0:
        return theta

    valid = True if observed is None else observed[interior]
    proportion = np.clip(np.mean(active_responses, axis=1, where=valid), 0.01, 0.99)
    estimates = np.clip(np.log(proportion / (1.0 - proportion)), -4.0, 4.0)
    lower = np.full(len(active_responses), -4.0)
    upper = np.full(len(active_responses), 4.0)
    active = np.ones(len(active_responses), dtype=bool)
    a_squared = a**2

    for _ in range(64):
        linear_predictor = a * (estimates[:, None] - b)
        probabilities = logistic_transform(linear_predictor)

        score = np.sum(a * (active_responses - probabilities), axis=1, where=valid)
        score_converged = active & (np.abs(score) <= 1e-12)
        active[score_converged] = False
        if not np.any(active):
            break

        positive = active & (score > 0.0)
        negative = active & ~positive
        lower[positive] = estimates[positive]
        upper[negative] = estimates[negative]

        information = np.sum(a_squared * probabilities * (1.0 - probabilities), axis=1, where=valid)
        candidates = estimates + np.divide(
            score,
            information,
            out=np.full(len(active_responses), np.nan),
            where=information > 0.0,
        )
        invalid = ~np.isfinite(candidates) | (candidates <= lower) | (candidates >= upper)
        candidates[invalid] = (lower[invalid] + upper[invalid]) / 2.0

        step_converged = active & (np.abs(candidates - estimates) <= 1e-12)
        estimates[active] = candidates[active]
        active[step_converged] = False
        if not np.any(active):
            break

    theta[interior] = estimates
    return theta


def _compute_lz(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, theta: np.ndarray, na_rm: bool = True
) -> np.ndarray:
    """Compute standardized log-likelihood in bounded respondent batches."""
    result = np.empty(len(x))
    batch_rows = max(1, _LZ_BATCH_ELEMENTS // x.shape[1])
    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        batch_result = np.full(stop - start, np.nan)
        batch_theta = theta[start:stop]
        valid = ~np.isnan(batch_theta)
        if np.any(valid):
            batch_result[valid] = _compute_lz_batch(
                x[start:stop][valid],
                a,
                b,
                batch_theta[valid],
                na_rm=na_rm,
            )
        result[start:stop] = batch_result
    return result


def _compute_lz_batch(
    responses: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    *,
    na_rm: bool = True,
) -> np.ndarray:
    """Compute lz scores using the observed items in one response batch."""
    observed = ~np.isnan(responses) if na_rm else None
    valid = True if observed is None or np.all(observed) else observed
    log_odds = np.asarray(a * (theta[:, None] - b), dtype=float)
    prob = logistic_transform(log_odds)
    np.clip(prob, _MIN_PROBABILITY, _MAX_PROBABILITY, out=prob)
    np.clip(log_odds, _MIN_LOG_ODDS, _MAX_LOG_ODDS, out=log_odds)

    # The centered Bernoulli identity avoids subtracting two nearly equal
    # log-likelihood sums. Retain the predictor instead of reconstructing its
    # log-odds from probabilities that can round to exactly 0.5.
    scales = np.max(np.abs(log_odds), axis=1, where=valid, initial=0.0)
    np.divide(log_odds, scales[:, None], out=log_odds, where=scales[:, None] != 0)
    log_odds[scales == 0] = 0.0
    # Common scaling cancels in lz and keeps tiny log-odds from losing variance
    # when squared. The same scratch matrix serves both moment reductions.
    scratch = responses - prob
    scratch *= log_odds
    centered_l = np.sum(scratch, axis=1, where=valid)
    np.square(log_odds, out=scratch)
    scratch *= prob
    scratch *= 1 - prob
    var_l = np.sum(scratch, axis=1, where=valid)
    result = np.where(np.isnan(var_l), np.nan, 0.0)
    np.divide(
        centered_l,
        np.sqrt(var_l),
        out=result,
        where=var_l > 0,
    )
    if observed is not None:
        result[~np.any(observed, axis=1)] = np.nan
    return result
