"""
Find and graph Mahalanobis Distance (D) and flag potential outliers

Takes a matrix of item responses and computes Mahalanobis D. Can additionally return a
vector of binary outlier flags.
Mahalanobis distance is calculated using a function which supports missing values.
The Mahalanobis distance is a measure of the distance between a point P and a distribution D,
introduced by P. C. Mahalanobis in 1936. It is a multi-dimensional generalization of the idea of
measuring how many standard deviations away P is from the mean of D. This distance is zero if P is
at the mean of D, and grows as P moves away from the mean along each principal component axis.
The Mahalanobis distance is thus unitless and scale-invariant, and takes into account the
correlations of the data set.
"""

from collections.abc import Iterator
from typing import Any, Literal, overload

import numpy as np

from ier._optional_imports import require_matplotlib_pyplot
from ier._row_statistics import row_slices
from ier._statistics import chi_square_quantile, chi_square_quantiles, normal_quantile
from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, validate_matrix_input, validate_probability


@overload
def mahad(
    x: MatrixLike,
    flag: Literal[False] = False,
    confidence: float = 0.95,
    na_rm: bool = False,
    method: str = "chi2",
) -> np.ndarray: ...


@overload
def mahad(
    x: MatrixLike,
    flag: Literal[True],
    confidence: float = 0.95,
    na_rm: bool = False,
    method: str = "chi2",
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def mahad(
    x: MatrixLike,
    flag: bool,
    confidence: float = 0.95,
    na_rm: bool = False,
    method: str = "chi2",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...


def mahad(
    x: MatrixLike,
    flag: bool = False,
    confidence: float = 0.95,
    na_rm: bool = False,
    method: str = "chi2",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Computes Mahalanobis Distance for a matrix of data.

    Mahalanobis distance measures how many standard deviations away a point is from the mean
    of a distribution, taking into account correlations between variables. It's useful for
    detecting multivariate outliers in survey data.

    Parameters:
    - x: Matrix of data where rows are observations and columns are variables.
          Any 2D array-like: nested lists or tuples, NumPy arrays, or DataFrames.
    - flag: If True, flags potential outliers based on the confidence level.
    - confidence: Confidence level for flagging outliers, a finite real number in
                  [0, 1]: a Python or NumPy scalar, 0-d array, ``Decimal`` or
                  ``Fraction``. Booleans and strings are rejected. Default is 0.95.
    - na_rm: If True, removes rows with missing data before computing distances,
             but reinserts NaNs in original positions. If False, raises error for missing data.
    - method: Method for outlier detection; every method flags only unusually large
              distances. Options: "chi2" (squared distance above the chi-squared
              quantile with one degree of freedom per variable), "iqr" (distance above
              the upper fence Q3 + 1.5 * IQR; ``confidence`` is not used) and "zscore"
              (standardized distance above the two-sided normal critical value for
              ``confidence``, e.g. 1.96 at 0.95).

    Returns:
    - Mahalanobis distances (with NaNs where removed), or
    - Tuple of (distances, flags) if `flag=True`.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid confidence, etc.)
    - TypeError: If input is a scalar, string, mapping, or other non-array object.
                 The exception also derives from ValueError.

    Example:
        >>> import numpy as np
        >>> data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
        ...         [1, 2], [2, 1], [4, 5], [5, 4], [5, 1]]
        >>> distances = mahad(data)
        >>> np.round(distances, 2).tolist()
        [1.4, 0.74, 0.28, 0.74, 1.4, 1.42, 1.11, 1.42, 1.11, 2.52]

        >>> distances, flags = mahad(data, flag=True, confidence=0.95)
        >>> flags.tolist()  # the last respondent contradicts the item correlation
        [False, False, False, False, False, False, False, False, False, True]
    """

    x_array = validate_matrix_input(x)
    confidence = validate_probability(confidence, name="confidence")

    if method not in ["chi2", "iqr", "zscore"]:
        raise ValueError("method must be one of: 'chi2', 'iqr', 'zscore'")

    valid_mask = np.empty(len(x_array), dtype=bool)
    for start, stop in row_slices(len(x_array), x_array.shape[1]):
        valid_mask[start:stop] = ~np.isnan(x_array[start:stop]).any(axis=1)
    n_valid = int(np.count_nonzero(valid_mask))
    has_missing = n_valid != len(x_array)
    if has_missing and not na_rm:
        raise ValueError("data contains missing values. Set na_rm=True to handle them")
    if n_valid == 0:
        raise ValueError("no complete cases found after removing missing values")
    if n_valid < 2:
        raise ValueError("at least two complete observations are required")
    if n_valid < x_array.shape[1]:
        raise ValueError(
            f"insufficient observations ({n_valid}) "
            f"for dimensions ({x_array.shape[1]}). "
            "Need more observations than variables."
        )

    distances = _compute_mahalanobis_distance(
        x_array, valid_mask=valid_mask if has_missing else None
    )

    if flag:
        flags = _flag_outliers(distances, confidence, method, x_array.shape[1])
        return distances, flags

    return distances


def _complete_blocks(
    x: np.ndarray,
    valid_mask: np.ndarray | None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield owned response blocks containing only complete observations."""
    for start, stop in row_slices(len(x), x.shape[1]):
        if valid_mask is None:
            block = np.array(x[start:stop], dtype=float, copy=True)
        else:
            valid = valid_mask[start:stop]
            if not np.any(valid):
                continue
            block = np.asarray(x[start:stop][valid], dtype=float)
        yield start, stop, block


def _compute_mahalanobis_distance(
    x: np.ndarray, *, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute Mahalanobis distances with respondent-bounded workspaces.

    Parameters:
    - x: Matrix of data (n_samples, n_features)
    - valid_mask: Optional mask selecting complete rows; excluded rows return NaN

    Returns:
    - Array of Mahalanobis distances
    """
    if valid_mask is None:
        n_valid = len(x)
        mean_vector = np.mean(x, axis=0, dtype=float)
    else:
        n_valid = int(np.count_nonzero(valid_mask))
        mean_vector = np.zeros(x.shape[1])
        for _, _, block in _complete_blocks(x, valid_mask):
            mean_vector += np.sum(block, axis=0)
        mean_vector /= n_valid
    cov_matrix = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for _, _, centered in _complete_blocks(x, valid_mask):
        np.subtract(centered, mean_vector, out=centered)
        cov_matrix += centered.T @ centered
    cov_matrix /= n_valid - 1

    u, s, vh = np.linalg.svd(cov_matrix, full_matrices=False, hermitian=True)

    eps = np.finfo(cov_matrix.dtype).eps
    # Preserve the inverse/pseudo-inverse cutoff without an overflowing ratio.
    threshold = 0.0 if s[-1] > eps * s[0] else eps * max(cov_matrix.shape) * s[0]
    inv_s = np.zeros_like(s)
    np.divide(1.0, s, out=inv_s, where=s > threshold)

    inv_cov_matrix = (vh.T * inv_s) @ u.T

    squared_distances = np.full(len(x), np.nan)
    for start, stop, centered in _complete_blocks(x, valid_mask):
        np.subtract(centered, mean_vector, out=centered)
        transformed = centered @ inv_cov_matrix
        block_distances = np.einsum("ij,ij->i", transformed, centered)
        if valid_mask is None:
            squared_distances[start:stop] = block_distances
        else:
            destination = squared_distances[start:stop]
            destination[valid_mask[start:stop]] = block_distances

    np.maximum(squared_distances, 0.0, out=squared_distances)
    np.sqrt(squared_distances, out=squared_distances)
    return squared_distances


def _flag_outliers(
    distances: np.ndarray, confidence: float, method: str, n_features: int
) -> np.ndarray:
    """
    Flag outliers based on Mahalanobis distances.

    Distance is an upper-tail measure, so only unusually large distances are
    flagged; distances near the multivariate center are never outliers.

    Parameters:
    - distances: Array of Mahalanobis distances, at least two of them available
    - confidence: Confidence level (0-1)
    - method: Outlier detection method, already validated by ``mahad``
    - n_features: Number of features (for chi2 degrees of freedom)

    Returns:
    - Boolean array indicating outliers
    """
    if method == "chi2":
        threshold = chi_square_quantile(confidence, n_features)
        result: np.ndarray = distances > np.sqrt(threshold)
        return result

    elif method == "iqr":
        valid_distances = distances[~np.isnan(distances)]
        q1, q3 = np.percentile(valid_distances, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        flags = np.full_like(distances, False, dtype=bool)
        valid_mask = ~np.isnan(distances)
        flags[valid_mask] = distances[valid_mask] > upper_bound
        return flags

    else:  # zscore
        valid_distances = distances[~np.isnan(distances)]
        mean_dist = np.mean(valid_distances)
        std_dist = np.std(valid_distances)

        if std_dist == 0:
            return np.full_like(distances, False, dtype=bool)

        # Keep the historical two-sided critical value, so the upper-tail cutoff
        # for a given confidence is unchanged; only the lower tail is dropped.
        z_threshold = normal_quantile(1 - (1 - confidence) / 2)

        flags = np.full_like(distances, False, dtype=bool)
        valid_mask = ~np.isnan(distances)
        z_scores = (distances[valid_mask] - mean_dist) / std_dist
        flags[valid_mask] = z_scores > z_threshold
        return flags


def mahad_qqplot(
    x: MatrixLike,
    na_rm: bool = False,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical and observed quantiles for a Mahalanobis distance Q-Q plot.

    Under the assumption of multivariate normality, squared Mahalanobis distances
    follow a chi-squared distribution with degrees of freedom equal to the number
    of variables. This function returns the quantiles needed to construct a Q-Q plot
    for assessing that assumption and identifying outliers.

    Parameters:
    - x: A matrix of data where rows are observations and columns are variables.
    - na_rm: If True, removes rows with missing data before computing distances.
    - plot: If True, renders a Q-Q plot using matplotlib (requires matplotlib).

    Returns:
    - Tuple of (theoretical_quantiles, observed_squared_distances), both sorted
      in ascending order. Theoretical quantiles are from the chi-squared distribution
      with p degrees of freedom.

    Raises:
    - RuntimeError: If plot=True and matplotlib is not available.
    - ValueError: If inputs are invalid.

    Example:
        >>> data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> theoretical, observed = mahad_qqplot(data)
    """
    x_array = validate_matrix_input(x)
    distances = mahad(x_array, na_rm=na_rm)

    valid_mask = ~np.isnan(distances)
    valid_distances = distances[valid_mask]

    observed_sq = np.sort(valid_distances**2)

    n = len(observed_sq)
    p = x_array.shape[1]

    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = chi_square_quantiles(probabilities, p)

    if plot:
        plt = require_matplotlib_pyplot()

        _fig, ax = plt.subplots(1, 1)
        ax.scatter(theoretical, observed_sq, edgecolors="black", facecolors="none")
        max_val = max(float(np.max(theoretical)), float(np.max(observed_sq)))
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1)
        ax.set_xlabel("Theoretical Chi-Squared Quantiles")
        ax.set_ylabel("Observed Squared Mahalanobis Distances")
        ax.set_title("Mahalanobis Distance Q-Q Plot")
        plt.show()

    return theoretical, observed_sq


def mahad_summary(x: MatrixLike, confidence: float = 0.95, na_rm: bool = False) -> dict[str, Any]:
    """
    Calculate summary statistics for Mahalanobis distances.

    Parameters:
    - x: Matrix of data where rows are observations and columns are variables
    - confidence: Confidence level for outlier detection
    - na_rm: If True, removes rows with missing data

    Returns:
    - Dictionary with distance summary statistics, the chi-squared outlier count
      (``outliers``), and respondent counts ``n_total``, ``n_valid`` and
      ``n_missing``. The same counts remain available as ``total``,
      ``valid_count`` and ``missing_count`` for compatibility.

    Example:
        >>> data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
        ...         [1, 2], [2, 1], [4, 5], [5, 4], [5, 1]]
        >>> summary = mahad_summary(data)
        >>> summary["outliers"], summary["n_total"], summary["n_valid"], summary["n_missing"]
        (1, 10, 10, 0)
        >>> round(summary["max"], 2)
        2.52
    """

    distances, flags = mahad(x, flag=True, confidence=confidence, na_rm=na_rm)

    valid_count = int(np.sum(~np.isnan(distances)))
    missing_count = len(distances) - valid_count
    stats = calculate_summary_stats(distances)
    stats.update(
        {
            "outliers": int(np.sum(flags)),
            "total": len(distances),
            "valid_count": valid_count,
            "missing_count": missing_count,
            "n_total": len(distances),
            "n_valid": valid_count,
            "n_missing": missing_count,
        }
    )
    return stats
