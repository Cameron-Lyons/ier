"""Validated loaders for reusable score vectors in versioned NumPy archives."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ier._registry import composite_index_names, validate_index_names
from ier._validation import validate_score_vectors

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.lib.npyio import NpzFile

    from ier.types import ScoreArchive, ScoreArchiveResultType

_SCORE_ARCHIVE_SCHEMA_VERSION = 1


def _require_member(archive: NpzFile, name: str) -> np.ndarray:
    """Load one required pickle-free member with a contextual error."""
    if name not in archive.files:
        raise ValueError(f"score archive is missing required member: {name}")
    try:
        value = archive[name]
    except ValueError as error:
        raise ValueError(f"score archive member {name} is not pickle-free") from error
    if not isinstance(value, np.ndarray):
        raise ValueError(f"score archive member {name} must be a NumPy array")
    return value


def _integer_scalar(archive: NpzFile, name: str) -> int:
    value = _require_member(archive, name)
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"score archive member {name} must be an integer scalar")
    return int(value.item())


def _string_scalar(archive: NpzFile, name: str) -> str:
    value = _require_member(archive, name)
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"score archive member {name} must be a Unicode string scalar")
    return str(value.item())


def _string_vector(archive: NpzFile, name: str) -> list[str]:
    value = _require_member(archive, name)
    if value.ndim != 1 or value.dtype.kind != "U":
        raise ValueError(f"score archive member {name} must be a Unicode string vector")
    return cast("list[str]", value.tolist())


def _load_errors(archive: NpzFile, score_names: set[str]) -> dict[str, str]:
    """Load aligned soft-failure metadata when present."""
    has_names = "error_names" in archive.files
    has_messages = "error_messages" in archive.files
    if has_names != has_messages:
        raise ValueError("score archive error names and messages must be stored together")
    if not has_names:
        return {}

    names = _string_vector(archive, "error_names")
    messages = _string_vector(archive, "error_messages")
    if len(names) != len(messages):
        raise ValueError("score archive error names and messages must have equal lengths")
    if len(names) != len(set(names)):
        raise ValueError("score archive error names must be unique")
    if score_names.intersection(names):
        raise ValueError("score archive indices cannot contain both scores and errors")
    validate_index_names(names)
    return dict(zip(names, messages, strict=True))


def _load_respondent_ids(archive: NpzFile, n_respondents: int) -> list[str] | None:
    """Load optional aligned respondent identifiers."""
    if "respondent_ids" not in archive.files:
        return None
    respondent_ids = _string_vector(archive, "respondent_ids")
    if len(respondent_ids) != n_respondents:
        raise ValueError("score archive respondent ID count must match n_respondents")
    if any(not value for value in respondent_ids):
        raise ValueError("score archive respondent IDs must be nonblank")
    if len(respondent_ids) != len(set(respondent_ids)):
        raise ValueError("score archive respondent IDs must be unique")
    return respondent_ids


def _load_score_members(
    archive: NpzFile,
    result_type: ScoreArchiveResultType,
    n_respondents: int,
) -> dict[str, np.ndarray]:
    """Load declared raw registered-index score members in archive order."""
    if "index_names" not in archive.files:
        if result_type == "composite":
            raise ValueError(
                "composite archive does not include component scores; "
                "write it with --include-components"
            )
        raise ValueError("screen archive is missing required member: index_names")

    names = _string_vector(archive, "index_names")
    if not names:
        raise ValueError("score archive does not contain reusable index scores")
    if any(not name for name in names):
        raise ValueError("score archive index names must be nonblank")
    if len(names) != len(set(names)):
        raise ValueError("score archive index names must be unique")
    validate_index_names(
        names,
        composite_index_names() if result_type == "composite" else None,
    )

    expected_members = {f"score__{name}" for name in names}
    actual_members = {name for name in archive.files if name.startswith("score__")}
    missing = expected_members - actual_members
    extra = actual_members - expected_members
    if missing:
        raise ValueError(f"score archive is missing declared score member: {min(missing)}")
    if extra:
        raise ValueError(f"score archive contains undeclared score member: {min(extra)}")

    raw_scores = {name: _require_member(archive, f"score__{name}") for name in names}
    scores, actual_respondents = validate_score_vectors(raw_scores)
    if actual_respondents != n_respondents:
        raise ValueError("score archive vectors must match n_respondents")
    return scores


def _read_score_archive(archive: NpzFile) -> ScoreArchive:
    """Validate one open NPZ archive and extract its reusable score payload."""
    if len(archive.files) != len(set(archive.files)):
        raise ValueError("score archive member names must be unique")

    schema_version = _integer_scalar(archive, "schema_version")
    if schema_version != _SCORE_ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported score archive schema version: {schema_version}; "
            f"expected {_SCORE_ARCHIVE_SCHEMA_VERSION}"
        )

    raw_result_type = _string_scalar(archive, "result_type")
    if raw_result_type not in {"screen", "composite"}:
        raise ValueError("score archive result_type must be 'screen' or 'composite'")
    result_type = cast("ScoreArchiveResultType", raw_result_type)

    n_respondents = _integer_scalar(archive, "n_respondents")
    if n_respondents < 1:
        raise ValueError("score archive n_respondents must be positive")

    scores = _load_score_members(archive, result_type, n_respondents)
    errors = _load_errors(archive, set(scores))
    respondent_ids = _load_respondent_ids(archive, n_respondents)
    return {
        "schema_version": schema_version,
        "result_type": result_type,
        "n_respondents": n_respondents,
        "scores": scores,
        "respondent_ids": respondent_ids,
        "errors": errors,
    }


def load_score_archive(path: str | Path) -> ScoreArchive:
    """
    Load reusable registered-index scores from a versioned CLI NPZ archive.

    The loader always disables pickling and validates schema version, result type,
    member names, registry membership, vector shape, respondent alignment,
    optional identifiers, and soft-failure metadata. Screen archives are reusable
    directly. Composite archives must have been written with
    ``--include-components`` so their raw public index scores are present.

    Parameters:
    - path: Path to a screen or detailed composite NPZ archive.

    Returns:
    - A ``ScoreArchive`` containing ordered raw score vectors, result metadata,
      optional respondent IDs, and any recorded per-index soft failures.

    Example:
        >>> from ier import composite_scores, load_score_archive, screen_scores
        >>> saved = load_score_archive("screening.npz")
        >>> updated_screen = screen_scores(saved["scores"], percentile=99)
        >>> saved_components = load_score_archive("composite.npz")
        >>> updated_composite = composite_scores(
        ...     saved_components["scores"],
        ...     weights={"irv": 2.0},
        ... )
    """
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        raise ValueError("score archive must be an NPZ archive")
    with loaded as archive:
        return _read_score_archive(archive)
