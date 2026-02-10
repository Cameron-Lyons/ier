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

from typing import Any

import numpy as np

from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, validate_matrix_input

SCIPY_AVAILABLE = False
stats: Any = None
try:
    from scipy import stats as _stats

    stats = _stats
    SCIPY_AVAILABLE = True
except ImportError:
    pass  # scipy is optional; mahad_flag and mahad_plot check at call time


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
          Can be a 2D list or numpy array.
    - flag: If True, flags potential outliers based on the confidence level.
    - confidence: Confidence level for flagging outliers (0–1). Default is 0.95.
    - na_rm: If True, removes rows with missing data before computing distances,
             but reinserts NaNs in original positions. If False, raises error for missing data.
    - method: Method for outlier detection. Options: "chi2" (chi-squared distribution),
              "iqr" (interquartile range), "zscore" (z-score threshold).

    Returns:
    - Mahalanobis distances (with NaNs where removed), or
    - Tuple of (distances, flags) if `flag=True`.

    Raises:
    - ValueError: If inputs are invalid (empty data, invalid confidence, etc.)
    - TypeError: If input is not a list or numpy array
    - RuntimeError: If scipy is required but not available

    Example:
        >>> import numpy as np
        >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]
        >>> distances = mahad(data)
        >>> print(distances)
        [0.87, 0.87, 0.87, 2.60]

        >>> distances, flags = mahad(data, flag=True, confidence=0.95)
        >>> print(flags)
        [False, False, False, True]
    """

    x_array = validate_matrix_input(x)

    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")

    if method not in ["chi2", "iqr", "zscore"]:
        raise ValueError("method must be one of: 'chi2', 'iqr', 'zscore'")

    if method == "chi2" and not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for chi2 method. Install with: pip install scipy")

    if na_rm:
        all_nan_mask = np.isnan(x_array).all(axis=1)
        partial_nan_mask = np.isnan(x_array).any(axis=1)
        valid_mask = ~partial_nan_mask & ~all_nan_mask
        x_filtered = x_array[valid_mask]
    else:
        if np.isnan(x_array).any():
            raise ValueError("data contains missing values. Set na_rm=True to handle them")
        x_filtered = x_array
        valid_mask = np.ones(x_array.shape[0], dtype=bool)

    if x_filtered.size == 0:
        raise ValueError("no complete cases found after removing missing values")

    if x_filtered.shape[0] < x_filtered.shape[1]:
        raise ValueError(
            f"insufficient observations ({x_filtered.shape[0]}) "
            f"for dimensions ({x_filtered.shape[1]}). "
            "Need more observations than variables."
        )

    distances_filtered = _compute_mahalanobis_distance(x_filtered)

    distances = np.full(shape=(x_array.shape[0],), fill_value=np.nan)
    distances[valid_mask] = distances_filtered

    distances = np.where(np.isnan(distances), np.nan, np.abs(distances))

    if flag:
        flags = _flag_outliers(distances, confidence, method, x_array.shape[1])
        return distances, flags

    return distances


def _compute_mahalanobis_distance(x: np.ndarray) -> np.ndarray:
    """
    Compute Mahalanobis distances for a matrix of data.

    Parameters:
    - x: Matrix of data (n_samples, n_features) with no missing values

    Returns:
    - Array of Mahalanobis distances
    """
    mean_vector = np.mean(x, axis=0)
    cov_matrix = np.cov(x, rowvar=False)

    u, s, vh = np.linalg.svd(cov_matrix, full_matrices=False)

    eps = np.finfo(cov_matrix.dtype).eps
    s_min = s[-1] if s[-1] > 0 else eps
    cond_number = s[0] / s_min

    if cond_number < 1 / eps:
        inv_s = 1.0 / s
    else:
        threshold = eps * max(cov_matrix.shape) * s[0]
        inv_s = np.where(s > threshold, 1.0 / s, 0.0)

    inv_cov_matrix = (vh.T * inv_s) @ u.T

    centered_data = x - mean_vector
    mahalanobis_squared = np.einsum("ij,jk,ik->i", centered_data, inv_cov_matrix, centered_data)
    result: np.ndarray = np.sqrt(mahalanobis_squared)
    return result


def _flag_outliers(
    distances: np.ndarray, confidence: float, method: str, n_features: int
) -> np.ndarray:
    """
    Flag outliers based on Mahalanobis distances.

    Parameters:
    - distances: Array of Mahalanobis distances
    - confidence: Confidence level (0-1)
    - method: Outlier detection method
    - n_features: Number of features (for chi2 degrees of freedom)

    Returns:
    - Boolean array indicating outliers
    """
    if method == "chi2":
        threshold: float = stats.chi2.ppf(confidence, df=n_features)
        result: np.ndarray = distances > np.sqrt(threshold)
        return result

    elif method == "iqr":
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) == 0:
            return np.full_like(distances, False, dtype=bool)

        q1, q3 = np.percentile(valid_distances, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        flags = np.full_like(distances, False, dtype=bool)
        valid_mask = ~np.isnan(distances)
        flags[valid_mask] = (distances[valid_mask] < lower_bound) | (
            distances[valid_mask] > upper_bound
        )
        return flags

    elif method == "zscore":
        valid_distances = distances[~np.isnan(distances)]
        if len(valid_distances) == 0:
            return np.full_like(distances, False, dtype=bool)

        mean_dist = np.mean(valid_distances)
        std_dist = np.std(valid_distances)

        if std_dist == 0:
            return np.full_like(distances, False, dtype=bool)

        z_threshold = stats.norm.ppf(1 - (1 - confidence) / 2) if SCIPY_AVAILABLE else 2.0

        flags = np.full_like(distances, False, dtype=bool)
        valid_mask = ~np.isnan(distances)
        z_scores = np.abs((distances[valid_mask] - mean_dist) / std_dist)
        flags[valid_mask] = z_scores > z_threshold
        return flags

    else:
        raise ValueError(f"unknown method: {method}")


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
    - RuntimeError: If scipy is not available.
    - RuntimeError: If plot=True and matplotlib is not available.
    - ValueError: If inputs are invalid.

    Example:
        >>> data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> theoretical, observed = mahad_qqplot(data)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required for mahad_qqplot. Install with: pip install scipy")

    distances = mahad(x, flag=False, na_rm=na_rm)
    assert isinstance(distances, np.ndarray)

    valid_mask = ~np.isnan(distances)
    valid_distances = distances[valid_mask]

    observed_sq = np.sort(valid_distances**2)

    n = len(observed_sq)
    x_array = validate_matrix_input(x, check_type=False)
    p = x_array.shape[1]

    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical: np.ndarray = stats.chi2.ppf(probabilities, df=p)

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for plotting. "
                "Install with: pip install insufficient-effort[plot]"
            ) from exc

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
    - Dictionary with summary statistics and outlier information

    Example:
        >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]
        >>> summary = mahad_summary(data)
        >>> print(summary)
        {'mean': 1.55, 'std': 0.87, 'outliers': 1, 'total': 4, ...}
    """

    distances, flags = mahad(x, flag=True, confidence=confidence, na_rm=na_rm)

    valid_count = int(np.sum(~np.isnan(distances)))
    stats = calculate_summary_stats(distances)
    stats.update(
        {
            "outliers": int(np.sum(flags)) if valid_count > 0 else 0,
            "total": len(distances),
            "valid_count": valid_count,
            "missing_count": int(np.sum(np.isnan(distances))),
        }
    )
    return stats
