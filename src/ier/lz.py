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
# The ability solver iterates only on unresolved rows, so larger batches
# amortize per-iteration overhead without recomputing converged rows.
_LZ_THETA_BATCH_ELEMENTS = 65_536
_MIN_PROBABILITY = 1e-10
_MAX_PROBABILITY = 1 - _MIN_PROBABILITY
_MIN_LOG_ODDS = math.log(_MIN_PROBABILITY / (1 - _MIN_PROBABILITY))
_MAX_LOG_ODDS = math.log(_MAX_PROBABILITY / (1 - _MAX_PROBABILITY))
_MIN_LOG_ODDS_EXACT = Fraction(_MIN_LOG_ODDS)
_MAX_LOG_ODDS_EXACT = Fraction(_MAX_LOG_ODDS)
# Away from zero, float64 spacing is at least 2**-533 at this boundary.
# Two parameters and a discrimination above it cannot form a subnormal product.
_SMALL_PARAMETER = 2.0**-480


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
        >>> np.round(lz_scores, 2).tolist()
        [0.19, 0.62, 0.1]
    """
    x_array = validate_matrix_input(x)

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
        >>> data = [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 1, 0, 1, 0]]
        >>> scores, flags = lz_flag(data, difficulty=[-2, -1, 0, 1, 2], model="1pl")
        >>> flags.tolist()
        [False, True, False]
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
    batch_rows = max(1, _LZ_THETA_BATCH_ELEMENTS // x.shape[1])
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
    magnitude = np.max(np.abs(a), where=~np.isnan(a), initial=0.0)
    scaled = magnitude > math.sqrt(np.finfo(float).max / len(a)) / 4
    if scaled:
        estimates = _initial_scaled_theta(active_responses, a, b, estimates, valid)

    # Iterate only on unresolved rows. A root beyond the bracket bisects toward
    # it for dozens of steps after the rest of the batch converges. Every row
    # reduction is independent, so dropping resolved rows from the workspace
    # leaves each estimate bit-identical.
    pending = np.flatnonzero(interior)
    lower = np.full(len(pending), -4.0)
    upper = np.full(len(pending), 4.0)
    # Saturated predictors and tiny moments are valid finite-model limits.
    # Unrepresentable Newton steps fall back to the safeguarded bracket.
    with np.errstate(over="ignore", under="ignore"):
        a_squared = None if scaled else a**2
        for _ in range(64):
            linear_predictor = a * (estimates[:, None] - b)
            probabilities = logistic_transform(linear_predictor)

            if scaled:
                score, steps, step_scales = _scaled_theta_steps(
                    active_responses, probabilities, a, valid
                )
                with np.errstate(divide="ignore"):
                    tolerance = 1e-12 / np.minimum(step_scales, 1.0)
            else:
                score = np.sum(a * (active_responses - probabilities), axis=1, where=valid)
                tolerance = 1e-12
            score_converged = np.abs(score) <= tolerance
            if np.all(score_converged):
                break

            positive = score > 0.0
            negative = ~positive
            lower[positive] = estimates[positive]
            upper[negative] = estimates[negative]

            if not scaled:
                assert a_squared is not None
                information = np.sum(
                    a_squared * probabilities * (1.0 - probabilities), axis=1, where=valid
                )
                steps = np.divide(
                    score,
                    information,
                    out=np.full(len(active_responses), np.nan),
                    where=information > 0.0,
                )
            candidates = estimates + steps
            invalid = ~np.isfinite(candidates) | (candidates <= lower) | (candidates >= upper)
            candidates[invalid] = (lower[invalid] + upper[invalid]) / 2.0

            changes = np.abs(candidates - estimates)
            if scaled:
                changes *= np.maximum(step_scales, 1.0)
            # Score-converged rows keep their current estimates; step-converged
            # rows keep their candidates. Float32 and float16 binary inputs start
            # in their own precision, so round each iterate back to it.
            np.copyto(candidates, estimates, where=score_converged)
            estimates = candidates.astype(estimates.dtype, copy=False)
            resolved = score_converged | (changes <= 1e-12)
            if np.any(resolved):
                theta[pending[resolved]] = estimates[resolved]
                unresolved = ~resolved
                pending = pending[unresolved]
                estimates = estimates[unresolved]
                if len(pending) == 0:
                    break
                lower = lower[unresolved]
                upper = upper[unresolved]
                active_responses = active_responses[unresolved]
                if not isinstance(valid, bool):
                    valid = valid[unresolved]

    # Score-converged rows and rows still moving after the iteration budget
    # keep their latest estimates.
    theta[pending] = estimates
    return theta


def _initial_scaled_theta(
    responses: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    logits: np.ndarray,
    valid: bool | np.ndarray,
) -> np.ndarray:
    """Start steep models near the response-weighted difficulty crossing."""
    magnitudes = np.abs(a)
    scales = np.max(np.broadcast_to(magnitudes, responses.shape), axis=1, where=valid, initial=0.0)
    with np.errstate(under="ignore", invalid="ignore"):
        weights = np.divide(
            magnitudes, scales[:, None], out=np.zeros(responses.shape), where=scales[:, None] != 0
        )
    if not isinstance(valid, bool):
        np.copyto(weights, 0.0, where=~valid)
    keyed_responses = np.where(a < 0, 1 - responses, responses)
    target = np.sum(weights * keyed_responses, axis=1, where=valid)
    order = np.argsort(b)
    cumulative = np.cumsum(weights[:, order], axis=1)
    crossing = np.argmax(cumulative >= target[:, None], axis=1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        estimates = b[order[crossing]] + np.divide(
            logits, scales, out=logits.copy(), where=scales != 0
        )
    result: np.ndarray = np.clip(estimates, -4.0, 4.0)
    return result


def _scaled_theta_steps(
    responses: np.ndarray, probabilities: np.ndarray, a: np.ndarray, valid: bool | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradient and Newton steps without squaring native discriminations."""
    residuals = responses - probabilities
    relevant = (residuals != 0) | ((probabilities > 0) & (probabilities < 1))
    relevant &= valid
    scales = np.max(
        np.broadcast_to(np.abs(a), responses.shape), axis=1, where=relevant, initial=0.0
    )
    with np.errstate(under="ignore", invalid="ignore"):
        normalized = np.divide(
            a,
            scales[:, None],
            out=np.zeros(responses.shape),
            where=relevant & (scales[:, None] != 0),
        )
        score = np.sum(normalized * residuals, axis=1, where=valid)
        information = np.sum(
            normalized**2 * probabilities * (1 - probabilities), axis=1, where=valid
        )
        # Separate exponents prevent score/information from overflowing before
        # division by the discrimination scale restores representable steps.
        score_mantissa, score_exponent = np.frexp(score)
        information_mantissa, information_exponent = np.frexp(information)
        scale_mantissa, scale_exponent = np.frexp(scales)
        steps = np.divide(
            score_mantissa,
            information_mantissa * scale_mantissa,
            out=np.full(len(responses), np.nan),
            where=(information > 0) & (scales > 0),
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        np.ldexp(steps, score_exponent - information_exponent - scale_exponent, out=steps)
    return score, steps, scales


def _compute_lz(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, theta: np.ndarray, na_rm: bool = True
) -> np.ndarray:
    """Compute standardized log-likelihood in bounded respondent batches."""
    result = np.empty(len(x))
    batch_rows = max(1, _LZ_BATCH_ELEMENTS // x.shape[1])
    parameter_extremes = _parameter_extremes(a, b, theta)
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
                parameter_extremes=parameter_extremes,
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
    parameter_extremes: tuple[bool, bool] | None = None,
) -> np.ndarray:
    """Compute lz scores using the observed items in one response batch."""
    observed = ~np.isnan(responses) if na_rm else None
    valid = True if observed is None or np.all(observed) else observed
    possible_overflow, possible_underflow = (
        _parameter_extremes(a, b, theta) if parameter_extremes is None else parameter_extremes
    )
    # Finite calibrated parameters can overflow their difference even when
    # multiplication by a small discrimination yields ordinary log-odds.
    # Products can also round to zero before the common lz scale cancels.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_odds = np.subtract(theta[:, None], b)
        overflowed_differences = np.isinf(log_odds) if possible_overflow else None
        np.multiply(log_odds, a, out=log_odds)
    if overflowed_differences is not None:
        for row, item in zip(*np.nonzero(overflowed_differences), strict=True):
            exact = _exact_log_odds(theta[row], a[item], b[item])
            if exact is not None:
                log_odds[row, item] = float(
                    min(max(exact, _MIN_LOG_ODDS_EXACT), _MAX_LOG_ODDS_EXACT)
                )
    repair_rows = None
    if possible_underflow:
        tiny_products = (np.abs(log_odds) < np.finfo(float).tiny) & (a != 0)
        tiny_products &= theta[:, None] != b
        repair_rows = np.any(tiny_products, axis=1)
    prob = logistic_transform(log_odds)
    np.clip(prob, _MIN_PROBABILITY, _MAX_PROBABILITY, out=prob)
    np.clip(log_odds, _MIN_LOG_ODDS, _MAX_LOG_ODDS, out=log_odds)

    # The centered Bernoulli identity avoids subtracting two nearly equal
    # log-likelihood sums. Retain the predictor instead of reconstructing its
    # log-odds from probabilities that can round to exactly 0.5.
    scales = np.max(np.abs(log_odds), axis=1, where=valid, initial=0.0)
    with np.errstate(under="ignore"):
        np.divide(log_odds, scales[:, None], out=log_odds, where=scales[:, None] != 0)
    log_odds[scales == 0] = 0.0
    if repair_rows is not None:
        for row in np.flatnonzero(repair_rows):
            _repair_scaled_log_odds(
                log_odds[row], a, b, theta[row], valid if isinstance(valid, bool) else valid[row]
            )
    # Common scaling cancels in lz and keeps tiny log-odds from losing variance
    # when squared. The same scratch matrix serves both moment reductions.
    with np.errstate(under="ignore"):
        scratch = responses - prob
        scratch *= log_odds
        centered_l = np.sum(scratch, axis=1, where=valid)
        np.square(log_odds, out=scratch)
        scratch *= prob
        scratch *= 1 - prob
    var_l = np.sum(scratch, axis=1, where=valid)
    result = np.where(np.isnan(var_l) | np.isnan(centered_l), np.nan, 0.0)
    np.divide(
        centered_l,
        np.sqrt(var_l),
        out=result,
        where=var_l > 0,
    )
    if observed is not None:
        result[~np.any(observed, axis=1)] = np.nan
    return result


def _parameter_extremes(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> tuple[bool, bool]:
    """Identify exceptional parameter units once for all respondent batches."""
    overflow = bool(np.any(np.abs(theta) > np.finfo(float).max / 2)) or bool(
        np.any(np.abs(b) > np.finfo(float).max / 2)
    )
    underflow = any(
        np.any((np.abs(values) < _SMALL_PARAMETER) & (values != 0)) for values in (a, b, theta)
    )
    return overflow, underflow


def _exact_log_odds(ability: float, discrimination: float, difficulty: float) -> Fraction | None:
    """Retain a finite calibrated predictor before exceptional float rounding."""
    if np.isnan(discrimination) or np.isnan(difficulty):
        return None
    return Fraction(float(discrimination)) * (
        Fraction(float(ability)) - Fraction(float(difficulty))
    )


def _repair_scaled_log_odds(
    destination: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ability: float,
    observed: bool | np.ndarray,
) -> None:
    """Normalize subnormal predictor products before converting them to floats."""
    predictors = [
        _exact_log_odds(ability, discrimination, difficulty)
        for discrimination, difficulty in zip(a, b, strict=True)
    ]
    available = np.full(len(a), observed, dtype=bool) if isinstance(observed, bool) else observed
    if any(
        predictor is None and present
        for predictor, present in zip(predictors, available, strict=True)
    ):
        destination.fill(np.nan)
        return
    clipped = [
        None if predictor is None else min(max(predictor, _MIN_LOG_ODDS_EXACT), _MAX_LOG_ODDS_EXACT)
        for predictor in predictors
    ]
    scale = max(
        (
            abs(predictor)
            for predictor, present in zip(clipped, available, strict=True)
            if present and predictor is not None
        ),
        default=Fraction(0),
    )
    for item, (predictor, present) in enumerate(zip(clipped, available, strict=True)):
        destination[item] = (
            np.nan if predictor is None else float(predictor / scale) if scale and present else 0.0
        )
