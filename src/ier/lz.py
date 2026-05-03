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

import logging

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input

logger = logging.getLogger(__name__)

SCIPY_AVAILABLE = False
try:
    from scipy import optimize

    SCIPY_AVAILABLE = True
except ImportError as e:
    # SciPy is optional; ML theta estimation falls back to logit transformation
    logger.debug("SciPy not available, ML theta estimation disabled: %s", e)


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
    - difficulty: Array of item difficulty parameters (b). If None, estimated from data.
    - discrimination: Array of item discrimination parameters (a). If None and model="2pl",
                     estimated from data. Ignored if model="1pl".
    - theta: Array of person ability estimates. If None, estimated from data.
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
        [0.12, -0.45, -2.31]
    """
    x_array = validate_matrix_input(x, check_type=False)

    if model not in ["1pl", "2pl"]:
        raise ValueError("model must be '1pl' or '2pl'")

    x_binary = _dichotomize(x_array)

    if difficulty is not None:
        b = np.asarray(difficulty)
        if len(b) != x_array.shape[1]:
            raise ValueError("difficulty length must match number of items")
    else:
        b = _estimate_difficulty(x_binary, na_rm=na_rm)

    if model == "2pl":
        if discrimination is not None:
            a = np.asarray(discrimination)
            if len(a) != x_array.shape[1]:
                raise ValueError("discrimination length must match number of items")
        else:
            a = _estimate_discrimination(x_binary, na_rm=na_rm)
    else:
        a = np.ones(x_array.shape[1])

    if theta is not None:
        theta_arr = np.asarray(theta)
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
    """Dichotomize polytomous responses at the observed midpoint."""
    unique_vals = np.unique(x[~np.isnan(x)])
    if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
        return x.copy()

    midpoint = (np.nanmax(x) + np.nanmin(x)) / 2
    result = np.where(np.isnan(x), np.nan, (x > midpoint).astype(float))
    return result


def _estimate_difficulty(x: np.ndarray, na_rm: bool = True) -> np.ndarray:
    """Estimate item difficulty from proportion correct."""
    p = np.nanmean(x, axis=0) if na_rm else np.mean(x, axis=0)
    p = np.clip(p, 0.001, 0.999)
    b: np.ndarray = -np.log(p / (1 - p))
    return b


def _estimate_discrimination(x: np.ndarray, na_rm: bool = True) -> np.ndarray:
    """Estimate item discrimination using point-biserial correlation."""
    n_items = x.shape[1]
    a = np.ones(n_items)

    total_score = np.nansum(x, axis=1) if na_rm else np.sum(x, axis=1)

    if np.std(total_score) == 0:
        return a

    for j in range(n_items):
        if na_rm:
            valid_mask = ~np.isnan(x[:, j])
            item_resp = x[valid_mask, j]
            scores = total_score[valid_mask]
        else:
            item_resp = x[:, j]
            scores = total_score

        if len(np.unique(item_resp)) < 2:
            continue

        if np.std(scores) == 0:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(item_resp, scores)
            r_pb = corr_matrix[0, 1]

        if np.isnan(r_pb):
            continue

        r_pb = np.clip(r_pb, -0.99, 0.99)
        a[j] = r_pb * 1.7 / np.sqrt(1 - r_pb**2)
        a[j] = np.clip(a[j], 0.2, 3.0)

    return a


def _estimate_theta(x: np.ndarray, a: np.ndarray, b: np.ndarray, na_rm: bool = True) -> np.ndarray:
    """Estimate person ability using ML or sum score transformation."""
    n_persons = x.shape[0]
    theta = np.zeros(n_persons)

    for i in range(n_persons):
        if na_rm:
            valid_mask = ~np.isnan(x[i, :])
            responses = x[i, valid_mask]
            a_valid = a[valid_mask]
            b_valid = b[valid_mask]
        else:
            responses = x[i, :]
            a_valid = a
            b_valid = b

        if len(responses) == 0:
            theta[i] = np.nan
            continue

        if np.all(responses == 1):
            theta[i] = 3.0
        elif np.all(responses == 0):
            theta[i] = -3.0
        elif SCIPY_AVAILABLE:
            theta[i] = _ml_theta(responses, a_valid, b_valid)
        else:
            p = np.mean(responses)
            p = np.clip(p, 0.01, 0.99)
            theta[i] = np.log(p / (1 - p))

    return theta


def _ml_theta(responses: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Maximum likelihood estimation of theta for a single person."""

    def neg_log_likelihood(theta_val: float) -> float:
        prob = 1 / (1 + np.exp(-a * (theta_val - b)))
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        ll: float = float(np.sum(responses * np.log(prob) + (1 - responses) * np.log(1 - prob)))
        return -ll

    result = optimize.minimize_scalar(neg_log_likelihood, bounds=(-4, 4), method="bounded")
    x_val: float = float(result.x)
    return x_val


def _compute_lz(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, theta: np.ndarray, na_rm: bool = True
) -> np.ndarray:
    """Compute standardized log-likelihood for each person."""
    n_persons = x.shape[0]
    lz_values = np.zeros(n_persons)

    for i in range(n_persons):
        if np.isnan(theta[i]):
            lz_values[i] = np.nan
            continue

        if na_rm:
            valid_mask = ~np.isnan(x[i, :])
            responses = x[i, valid_mask]
            a_valid = a[valid_mask]
            b_valid = b[valid_mask]
        else:
            responses = x[i, :]
            a_valid = a
            b_valid = b

        if len(responses) == 0:
            lz_values[i] = np.nan
            continue

        prob = 1 / (1 + np.exp(-a_valid * (theta[i] - b_valid)))
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        log_l = np.sum(responses * np.log(prob) + (1 - responses) * np.log(1 - prob))

        expected_l = np.sum(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))

        log_odds = np.log(prob / (1 - prob))
        var_l = np.sum(prob * (1 - prob) * log_odds**2)

        if var_l <= 0:
            lz_values[i] = 0.0
        else:
            lz_values[i] = (log_l - expected_l) / np.sqrt(var_l)

    return lz_values
