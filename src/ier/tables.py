"""
Respondent-aligned result tables and index agreement matrices.

Tables are ordered mappings of column names to the existing result arrays, so
they pass directly to ``pandas.DataFrame`` or ``polars.DataFrame`` without
copying scores or adding a dependency. Agreement matrices compare how often
indices flag the same respondents or how closely their scores rank them.
"""

from collections.abc import Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from itertools import repeat

import numpy as np

from ier._column_statistics import column_correlations
from ier._registry import INDEX_REGISTRY
from ier._row_statistics import row_slices
from ier.types import AgreementKind, CompositeSummary, FloatArray, ScreenResult

_AGREEMENT_KINDS = ("overlap", "jaccard", "spearman")
# Containers without a defined respondent order, or consumed when read.
_UNORDERED_ID_CONTAINERS: tuple[tuple[type, str], ...] = (
    (AbstractSet, "a set"),
    (Mapping, "a mapping"),
    (Iterator, "an iterator"),
)


def _validate_flag(value: bool, name: str) -> bool:
    """Return a validated Boolean column-selection control."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _respondent_id_values(
    respondent_ids: Sequence[str], n_respondents: int, *, prefix: str = ""
) -> list[str]:
    """Return aligned string identifiers that fixed-width NumPy Unicode stores exactly.

    Tables and archive writers share this check so they raise the same exception
    types. Sets, mappings, and iterators are rejected because they carry no
    respondent order (or are consumed by reading), so their identifiers could
    silently attach to the wrong respondents. Sequences, one-dimensional NumPy
    arrays, and other ordered, re-iterable containers such as a pandas Index are
    accepted. NumPy Unicode arrays silently drop a trailing NUL, merging IDs such
    as ``"r1"`` and ``"r1\\0"``, so such identifiers are rejected.
    """
    candidate: object = respondent_ids
    if isinstance(candidate, (str, bytes)):
        raise TypeError("respondent_ids must be a sequence of strings, not a single string")
    for unordered, label in _UNORDERED_ID_CONTAINERS:
        if isinstance(candidate, unordered):
            raise TypeError(
                f"respondent_ids must be a sequence of strings in respondent order, not {label}"
            )
    try:
        values = list(respondent_ids)
    except TypeError as error:
        raise TypeError("respondent_ids must be a sequence of strings") from error
    if len(values) != n_respondents:
        raise ValueError(
            f"{prefix}respondent ID count must match n_respondents: "
            f"expected {n_respondents}, received {len(values)}"
        )
    if not all(map(isinstance, values, repeat(str))):
        raise ValueError(f"{prefix}respondent IDs must be strings")
    if any(map(str.endswith, values, repeat("\0"))):
        raise ValueError(f"{prefix}respondent IDs cannot end with a NUL character")
    return values


def _respondent_column(respondent_ids: Sequence[str], n_respondents: int) -> np.ndarray:
    """Validate aligned string identifiers and return them as a NumPy string column."""
    return np.asarray(_respondent_id_values(respondent_ids, n_respondents), dtype=np.str_)


def _aligned_table(
    columns: Mapping[str, object],
    n_respondents: int,
    respondent_ids: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    """Collect respondent-aligned columns by reference after checking their lengths."""
    table: dict[str, np.ndarray] = {}
    if respondent_ids is not None:
        table["respondent"] = _respondent_column(respondent_ids, n_respondents)
    for name, values in columns.items():
        column = np.asarray(values)
        if column.shape != (n_respondents,):
            raise ValueError(f"result column {name} must contain one value per respondent")
        table[name] = column
    return table


def screen_table(
    screen_result: ScreenResult,
    *,
    respondent_ids: Sequence[str] | None = None,
    include_scores: bool = True,
    include_flags: bool = True,
) -> dict[str, np.ndarray]:
    """
    Arrange a screening result as respondent-aligned columns.

    Columns follow the CLI CSV schema without its ``respondent`` label column:
    ``flag_count``, ``valid_index_count``, ``consensus_eligible``, and
    ``consensus_flag``, then ``{name}_score`` and ``{name}_flag`` for each index
    in ``screen_result["indices_used"]``. Arrays are returned by reference, never copied,
    so the mapping is cheap to build and should not be mutated in place.

    Parameters:
    - screen_result: Output dict from screen() or screen_scores().
    - respondent_ids: Optional identifiers, one string per respondent in
      respondent order, with no trailing NUL character, given as a sequence or
      one-dimensional array. When given, they form a leading ``respondent``
      column of NumPy strings.
    - include_scores: Include each index's score column.
    - include_flags: Include each index's flag column.

    Returns:
    - Ordered dict of column names to one-dimensional NumPy arrays. Pass it to
      ``pandas.DataFrame(table, index=df.index)`` to restore a DataFrame index.

    Raises:
    - TypeError: If respondent_ids is a single string, is not iterable, or is a
      set, mapping, or iterator, none of which has a defined respondent order.
    - ValueError: If identifiers are misaligned, not strings, or end with a NUL
      character, a column has the wrong length, or an include option is not a
      boolean.

    Example:
        >>> from ier import screen_scores, screen_table
        >>> result = screen_scores(
        ...     {"irv": [0.1, 0.9, 1.2], "longstring": [8.0, 2.0, 3.0]},
        ...     thresholds={"irv": 0.5, "longstring": 5.0},
        ... )
        >>> table = screen_table(result, respondent_ids=["r1", "r2", "r3"])
        >>> list(table)
        ['respondent', 'flag_count', 'valid_index_count', 'consensus_eligible',
         'consensus_flag', 'irv_score', 'irv_flag', 'longstring_score', 'longstring_flag']
        >>> table["flag_count"].tolist()
        [2, 0, 0]
        >>> scores = result["scores"]
        >>> table["irv_score"] is scores["irv"]
        True
    """
    include_scores = _validate_flag(include_scores, "include_scores")
    include_flags = _validate_flag(include_flags, "include_flags")
    columns: dict[str, object] = {
        "flag_count": screen_result["flag_counts"],
        "valid_index_count": screen_result["valid_index_counts"],
        "consensus_eligible": screen_result["consensus_eligible"],
        "consensus_flag": screen_result["consensus_flags"],
    }
    for name in screen_result["indices_used"]:
        if include_scores:
            columns[f"{name}_score"] = screen_result["scores"][name]
        if include_flags:
            columns[f"{name}_flag"] = screen_result["flags"][name]
    return _aligned_table(columns, screen_result["n_respondents"], respondent_ids)


def composite_table(
    summary: CompositeSummary,
    *,
    respondent_ids: Sequence[str] | None = None,
    include_components: bool = True,
) -> dict[str, np.ndarray]:
    """
    Arrange a detailed composite result as respondent-aligned columns.

    Columns are ``composite_score`` and ``valid_index_count``, followed by
    ``{name}_score`` for each successfully scored component index, matching the
    CLI composite CSV written with ``--include-components``. Arrays are returned
    by reference, never copied.

    Parameters:
    - summary: Output dict from composite_summary() or composite_scores_summary().
    - respondent_ids: Optional identifiers, one string per respondent in
      respondent order, with no trailing NUL character, given as a sequence or
      one-dimensional array. When given, they form a leading ``respondent``
      column of NumPy strings.
    - include_components: Include each raw component index score column.

    Returns:
    - Ordered dict of column names to one-dimensional NumPy arrays.

    Raises:
    - TypeError: If respondent_ids is a single string, is not iterable, or is a
      set, mapping, or iterator, none of which has a defined respondent order.
    - ValueError: If identifiers are misaligned, not strings, or end with a NUL
      character, a column has the wrong length, or include_components is not a
      boolean.

    Example:
        >>> from ier import composite_scores_summary, composite_table
        >>> summary = composite_scores_summary(
        ...     {"irv": [0.2, 1.0, 1.4], "longstring": [6.0, 2.0, 1.0]}
        ... )
        >>> table = composite_table(summary)
        >>> list(table)
        ['composite_score', 'valid_index_count', 'irv_score', 'longstring_score']
        >>> table["valid_index_count"].tolist()
        [2, 2, 2]
    """
    include_components = _validate_flag(include_components, "include_components")
    columns: dict[str, object] = {
        "composite_score": summary["composite"],
        "valid_index_count": summary["valid_index_counts"],
    }
    if include_components:
        for name, scores in summary["indices"].items():
            columns[f"{name}_score"] = scores
    return _aligned_table(columns, summary["n_total"], respondent_ids)


def _average_ranks(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Assign midranks to tied values and report whether the ranks vary."""
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2
    return midranks[inverse.reshape(-1)], len(unique) > 1


def _coflag_counts(flags: Sequence[np.ndarray], n_respondents: int) -> FloatArray:
    """Count respondents flagged by each index pair in bounded row batches.

    Floating-point products of zero/one indicators accumulate exact integer
    counts below 2**53 while using optimized matrix multiplication.
    """
    k = len(flags)
    counts = np.zeros((k, k))
    if k == 0:
        return counts
    for start, stop in row_slices(n_respondents, k):
        block = np.empty((stop - start, k))
        for column, values in enumerate(flags):
            block[:, column] = values[start:stop]
        counts += block.T @ block
    return counts


def _spearman_matrix(scores: Sequence[np.ndarray]) -> FloatArray:
    """Correlate average-tie score ranks over pairwise-complete respondents.

    Indices with identical finite-score masks are ranked once and correlated
    together; pairs with different availability are re-ranked on their shared
    respondents. Fewer than three shared scores or constant ranks give NaN.
    """
    k = len(scores)
    correlations = np.full((k, k), np.nan)
    masks = [np.isfinite(values) for values in scores]
    counts = [int(np.count_nonzero(mask)) for mask in masks]

    groups: list[list[int]] = []
    for position, mask in enumerate(masks):
        for group in groups:
            first = group[0]
            if counts[first] == counts[position] and np.array_equal(masks[first], mask):
                group.append(position)
                break
        else:
            groups.append([position])

    ranks: dict[int, np.ndarray] = {}
    group_of = [0] * k
    for group_number, group in enumerate(groups):
        for position in group:
            group_of[position] = group_number
        n_shared = counts[group[0]]
        if n_shared < 3:
            continue
        # Column-major storage keeps each reusable rank vector contiguous.
        stacked = np.empty((n_shared, len(group)), order="F")
        defined = []
        for column, position in enumerate(group):
            column_ranks, varies = _average_ranks(scores[position][masks[position]])
            stacked[:, column] = column_ranks
            ranks[position] = stacked[:, column]
            defined.append(varies)
        members = np.asarray(group)
        if len(group) > 1:
            # Mirror one triangle so separately rounded normalizations stay symmetric.
            rows, columns = np.triu_indices(len(group), 1)
            upper = column_correlations(stacked)[rows, columns]
            correlations[members[rows], members[columns]] = upper
            correlations[members[columns], members[rows]] = upper
        correlations[members, members] = np.where(defined, 1.0, np.nan)

    def shared_ranks(position: int, shared: np.ndarray, n_shared: int) -> np.ndarray:
        if n_shared == counts[position]:
            return ranks[position]
        return _average_ranks(scores[position][shared])[0]

    for left in range(k):
        for right in range(left + 1, k):
            if group_of[left] == group_of[right]:
                continue
            shared = masks[left] & masks[right]
            n_shared = int(np.count_nonzero(shared))
            if n_shared < 3:
                continue
            pair = np.column_stack(
                (shared_ranks(left, shared, n_shared), shared_ranks(right, shared, n_shared))
            )
            correlations[left, right] = correlations[right, left] = column_correlations(pair)[0, 1]
    return correlations


def index_agreement(
    screen_result: ScreenResult,
    kind: AgreementKind = "jaccard",
) -> tuple[list[str], FloatArray]:
    """
    Compare screening indices by shared flags or score rankings.

    Kinds:
    - ``"overlap"``: number of respondents flagged by both indices. The diagonal
      is each index's flagged count.
    - ``"jaccard"``: shared flags divided by respondents flagged by either index,
      or NaN when neither index flags anyone.
    - ``"spearman"``: rank correlation of suspiciousness over respondents with
      finite scores on both indices, using average ranks for ties. Scores of
      indices that flag low values, such as irv and psychsyn, are negated before
      ranking, so a positive correlation always means both indices rank the same
      respondents as more suspicious. Unregistered score names keep their raw
      orientation. Pairs with fewer than three shared scores or constant ranks
      are NaN. Presence-flagged indices such as onset are excluded because their
      scores are event positions.

    Parameters:
    - screen_result: Output dict from screen() or screen_scores().
    - kind: Agreement measure.

    Returns:
    - Tuple of (index_names, matrix) where matrix is a symmetric float array
      ordered like ``index_names``.

    Raises:
    - ValueError: If kind is not recognized.

    Example:
        >>> from ier import index_agreement, screen_scores
        >>> result = screen_scores(
        ...     {"irv": [0.1, 0.2, 0.9, 1.2], "longstring": [8.0, 3.0, 2.0, 7.0]},
        ...     thresholds={"irv": 0.5, "longstring": 5.0},
        ... )
        >>> names, overlap = index_agreement(result, kind="overlap")
        >>> names
        ['irv', 'longstring']
        >>> overlap.tolist()
        [[2.0, 1.0], [1.0, 2.0]]
        >>> index_agreement(result)[1].round(2).tolist()
        [[1.0, 0.33], [0.33, 1.0]]

        Low irv and high longstring scores are both suspicious, so respondents
        ranked similarly by the two indices give a positive correlation:

        >>> index_agreement(result, kind="spearman")[1].round(2).tolist()
        [[1.0, 0.4], [0.4, 1.0]]
    """
    if not isinstance(kind, str) or kind not in _AGREEMENT_KINDS:
        raise ValueError("kind must be 'overlap', 'jaccard', or 'spearman'")
    names = list(screen_result["indices_used"])
    n_respondents = screen_result["n_respondents"]
    if kind == "spearman":
        names = [
            name
            for name in names
            if name not in INDEX_REGISTRY or INDEX_REGISTRY[name].flag_mode != "present"
        ]
        vectors = [np.asarray(screen_result["scores"][name]) for name in names]
    else:
        vectors = [np.asarray(screen_result["flags"][name], dtype=bool) for name in names]
    for name, values in zip(names, vectors, strict=True):
        if values.shape != (n_respondents,):
            raise ValueError(f"{name} result vector must contain one value per respondent")
    if kind == "spearman":
        # Negating scores reverses their average ranks exactly, so orienting the
        # low-direction indices only flips the sign of their correlations.
        signs = np.asarray(
            [
                -1.0
                if name in INDEX_REGISTRY and INDEX_REGISTRY[name].flag_direction == "low"
                else 1.0
                for name in names
            ]
        )
        return names, _spearman_matrix(vectors) * np.outer(signs, signs)

    counts = _coflag_counts(vectors, n_respondents)
    if kind == "overlap":
        return names, counts
    flagged = counts.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = counts / (flagged[:, None] + flagged[None, :] - counts)
    return names, jaccard
