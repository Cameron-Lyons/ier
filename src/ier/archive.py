"""Validated persistence for reusable results in versioned NumPy archives."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from stat import S_IMODE
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from ier._flagging import resolve_threshold, validate_percentile
from ier._registry import composite_index_names, validate_index_names
from ier._validation import validate_score_array, validate_score_vectors
from ier.psychsyn import PsychsynModel
from ier.response_time import ResponseTimeMixtureModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.lib.npyio import NpzFile
    from numpy.typing import ArrayLike

    from ier.types import (
        BoolArray,
        InspectableArchive,
        ResponseTimeArchive,
        ResponseTimeFlagDirection,
        ResponseTimeMetric,
        ResponseTimeThresholdSource,
        ScoreArchive,
        ScoreArchiveResultType,
    )

_ARCHIVE_SCHEMA_VERSION = 1
_RESPONSE_TIME_ARCHIVE_SCHEMA_VERSION = 2
_RESPONSE_TIME_MIXTURE_MODEL_SCHEMA_VERSION = 1
_PSYCHSYN_MODEL_SCHEMA_VERSION = 1


def _validate_result_type(value: object) -> ScoreArchiveResultType:
    """Return a supported reusable-score archive result type."""
    if not isinstance(value, str) or value not in {"screen", "composite"}:
        raise ValueError("score archive result_type must be 'screen' or 'composite'")
    return cast("ScoreArchiveResultType", value)


def _validate_archive_index_names(
    names: list[str],
    result_type: ScoreArchiveResultType,
    *,
    allow_empty: bool = False,
    label: str = "index",
) -> None:
    """Validate ordered score or error names against the relevant registry."""
    if not names and not allow_empty:
        raise ValueError("score archive does not contain reusable index scores")
    if any(not isinstance(name, str) for name in names):
        raise ValueError(f"score archive {label} names must be strings")
    if any(not name.strip() for name in names):
        raise ValueError(f"score archive {label} names must be nonblank")
    if len(names) != len(set(names)):
        raise ValueError(f"score archive {label} names must be unique")
    validate_index_names(
        names,
        composite_index_names() if result_type == "composite" else None,
    )


def _validate_error_metadata(
    names: list[str],
    messages: list[str],
    score_names: set[str],
    result_type: ScoreArchiveResultType,
) -> dict[str, str]:
    """Validate aligned soft-failure metadata and preserve its order."""
    if len(names) != len(messages):
        raise ValueError("score archive error names and messages must have equal lengths")
    _validate_archive_index_names(names, result_type, allow_empty=True, label="error")
    if score_names.intersection(names):
        raise ValueError("score archive indices cannot contain both scores and errors")
    if any(not isinstance(message, str) for message in messages):
        raise ValueError("score archive error messages must be strings")
    if any(not message.strip() for message in messages):
        raise ValueError("score archive error messages must be nonblank")
    return dict(zip(names, messages, strict=True))


def _validate_respondent_ids(values: list[str], n_respondents: int) -> list[str]:
    """Validate aligned, unique, nonblank respondent identifiers."""
    if len(values) != n_respondents:
        raise ValueError("archive respondent ID count must match n_respondents")
    if any(not isinstance(value, str) for value in values):
        raise ValueError("archive respondent IDs must be strings")
    if any(not value.strip() for value in values):
        raise ValueError("archive respondent IDs must be nonblank")
    if len(values) != len(set(values)):
        raise ValueError("archive respondent IDs must be unique")
    return values


def _stream_npz_archive(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Stream typed arrays directly into one uncompressed NPZ archive."""
    with ZipFile(path, mode="w", compression=ZIP_STORED, allowZip64=True) as archive:
        for name, value in payload.items():
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                np.save(member, value, allow_pickle=False)


def _write_npz_archive(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Atomically stream one typed, pickle-free NPZ archive into place."""
    if any(value.dtype.hasobject for value in payload.values()):
        raise ValueError("NPZ archive cannot contain object arrays")

    destination = path.resolve(strict=False) if path.is_symlink() else path
    existing_mode: int | None = None
    with suppress(FileNotFoundError):
        existing_mode = S_IMODE(destination.stat().st_mode)

    with TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as directory:
        staged_path = Path(directory) / destination.name
        _stream_npz_archive(staged_path, payload)
        if existing_mode is not None:
            staged_path.chmod(existing_mode)
        staged_path.replace(destination)


def _require_member(archive: NpzFile, name: str) -> np.ndarray:
    """Load one required pickle-free member with a contextual error."""
    if name not in archive.files:
        raise ValueError(f"NPZ archive is missing required member: {name}")
    try:
        value = archive[name]
    except ValueError as error:
        raise ValueError(f"NPZ archive member {name} is not pickle-free") from error
    if not isinstance(value, np.ndarray):
        raise ValueError(f"NPZ archive member {name} must be a NumPy array")
    return value


def _open_npz_archive(path: str | Path, *, label: str) -> NpzFile:
    """Open one pickle-disabled NPZ archive with a contextual container error."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        raise ValueError(f"{label} archive must be an NPZ archive")
    return cast("NpzFile", loaded)


def _integer_scalar(archive: NpzFile, name: str) -> int:
    value = _require_member(archive, name)
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"NPZ archive member {name} must be an integer scalar")
    return int(value.item())


def _string_scalar(archive: NpzFile, name: str) -> str:
    value = _require_member(archive, name)
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"NPZ archive member {name} must be a Unicode string scalar")
    return str(value.item())


def _string_vector(archive: NpzFile, name: str) -> list[str]:
    value = _require_member(archive, name)
    if value.ndim != 1 or value.dtype.kind != "U":
        raise ValueError(f"NPZ archive member {name} must be a Unicode string vector")
    return cast("list[str]", value.tolist())


def _numeric_scalar(archive: NpzFile, name: str) -> float:
    """Load one finite real numeric scalar from an archive."""
    return _finite_numeric_scalar(
        _require_member(archive, name),
        name=f"NPZ archive member {name}",
    )


def _boolean_scalar(archive: NpzFile, name: str) -> bool:
    """Load one strict Boolean scalar from an archive."""
    value = _require_member(archive, name)
    if value.shape != () or value.dtype.kind != "b":
        raise ValueError(f"NPZ archive member {name} must be a Boolean scalar")
    return bool(value.item())


def _numeric_vector(archive: NpzFile, name: str) -> np.ndarray:
    """Load one real numeric vector from an archive."""
    value = _require_member(archive, name)
    if value.ndim != 1 or value.dtype.kind not in "fiu":
        raise ValueError(f"NPZ archive member {name} must be a real numeric vector")
    return np.asarray(value, dtype=float)


def _finite_numeric_scalar(value: object, *, name: str) -> float:
    """Validate and return one finite real numeric scalar."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric scalar") from error
    if array.shape != () or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_boolean_vector(
    values: ArrayLike,
    n_respondents: int,
    *,
    name: str,
) -> BoolArray:
    """Validate one aligned one-dimensional boolean vector."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a boolean vector") from error
    if array.ndim != 1 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean vector")
    if len(array) != n_respondents:
        raise ValueError(f"{name} must match n_respondents")
    return cast("BoolArray", array)


def _load_errors(
    archive: NpzFile,
    score_names: set[str],
    result_type: ScoreArchiveResultType,
) -> dict[str, str]:
    """Load aligned soft-failure metadata when present."""
    has_names = "error_names" in archive.files
    has_messages = "error_messages" in archive.files
    if has_names != has_messages:
        raise ValueError("score archive error names and messages must be stored together")
    if not has_names:
        return {}

    names = _string_vector(archive, "error_names")
    messages = _string_vector(archive, "error_messages")
    return _validate_error_metadata(names, messages, score_names, result_type)


def _load_respondent_ids(archive: NpzFile, n_respondents: int) -> list[str] | None:
    """Load optional aligned respondent identifiers."""
    if "respondent_ids" not in archive.files:
        return None
    respondent_ids = _string_vector(archive, "respondent_ids")
    return _validate_respondent_ids(respondent_ids, n_respondents)


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
    _validate_archive_index_names(names, result_type)

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
    if schema_version != _ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported score archive schema version: {schema_version}; "
            f"expected {_ARCHIVE_SCHEMA_VERSION}"
        )

    result_type = _validate_result_type(_string_scalar(archive, "result_type"))

    n_respondents = _integer_scalar(archive, "n_respondents")
    if n_respondents < 1:
        raise ValueError("score archive n_respondents must be positive")

    scores = _load_score_members(archive, result_type, n_respondents)
    errors = _load_errors(archive, set(scores), result_type)
    respondent_ids = _load_respondent_ids(archive, n_respondents)
    return {
        "schema_version": schema_version,
        "result_type": result_type,
        "n_respondents": n_respondents,
        "scores": scores,
        "respondent_ids": respondent_ids,
        "errors": errors,
    }


def _validate_response_time_metric(value: object) -> ResponseTimeMetric:
    """Return a supported archived response-time metric."""
    if not isinstance(value, str) or value not in {
        "mean",
        "median",
        "sd",
        "min",
        "consistency",
        "mixture",
    }:
        raise ValueError("response-time archive contains an unsupported metric")
    return cast("ResponseTimeMetric", value)


def _validate_response_time_direction(value: object) -> ResponseTimeFlagDirection:
    """Return a supported archived response-time flag direction."""
    if not isinstance(value, str) or value not in {"high", "low"}:
        raise ValueError("response-time archive flag_direction must be 'high' or 'low'")
    return cast("ResponseTimeFlagDirection", value)


def _validate_response_time_threshold_source(
    value: object,
) -> ResponseTimeThresholdSource:
    """Return a supported response-time threshold provenance value."""
    if not isinstance(value, str) or value not in {"fixed", "percentile"}:
        raise ValueError("response-time archive threshold_source must be 'fixed' or 'percentile'")
    return cast("ResponseTimeThresholdSource", value)


def _validate_response_time_percentile(
    value: object,
    source: ResponseTimeThresholdSource,
) -> float | None:
    """Validate the percentile field required by response-time schema v2."""
    if value is None:
        if source == "percentile":
            raise ValueError(
                "response-time archive percentile is required for percentile thresholds"
            )
        return None
    if source == "fixed":
        raise ValueError("response-time archive percentile must be absent for fixed thresholds")
    return validate_percentile(value)  # type: ignore[arg-type]


def _optional_percentile_scalar(archive: NpzFile, name: str) -> float | None:
    """Load a numeric percentile scalar, treating NaN as an absent value."""
    value = _require_member(archive, name)
    if value.shape != () or value.dtype.kind not in "fiu":
        raise ValueError(f"NPZ archive member {name} must be a numeric scalar")
    result = float(value.item())
    return None if np.isnan(result) else result


def _validate_response_time_values(
    scores: ArrayLike,
    flags: ArrayLike,
    metric: object,
    direction: object,
    threshold: object,
    *,
    expected_respondents: int | None = None,
    threshold_source: ResponseTimeThresholdSource | None = None,
    percentile: float | None = None,
) -> tuple[
    np.ndarray,
    BoolArray,
    ResponseTimeMetric,
    ResponseTimeFlagDirection,
    float,
]:
    """Validate one complete reusable response-time result."""
    validated_metric = _validate_response_time_metric(metric)
    validated_direction = _validate_response_time_direction(direction)
    expected_direction = "high" if validated_metric == "mixture" else "low"
    if validated_direction != expected_direction:
        raise ValueError(
            f"response-time metric {validated_metric!r} requires "
            f"{expected_direction!r} flag_direction"
        )

    validated_threshold = _finite_numeric_scalar(
        threshold,
        name="response-time archive threshold",
    )
    validated_scores = validate_score_array(scores, name="response-time archive scores")
    if expected_respondents is not None and len(validated_scores) != expected_respondents:
        raise ValueError("response-time archive scores must match n_respondents")
    validated_flags = _validate_boolean_vector(
        flags,
        len(validated_scores),
        name="response-time archive flags",
    )

    validated_percentile = (
        None
        if threshold_source is None
        else _validate_response_time_percentile(percentile, threshold_source)
    )
    if threshold_source == "percentile":
        assert validated_percentile is not None
        resolved_threshold = resolve_threshold(
            validated_scores,
            None,
            validated_percentile,
        )
        if not np.isclose(
            validated_threshold,
            resolved_threshold,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("response-time archive threshold is inconsistent with its percentile")

    expected_flags = np.empty_like(validated_flags)
    inclusive_compare = np.greater_equal if validated_direction == "high" else np.less_equal
    exclusive_compare = np.greater if validated_direction == "high" else np.less
    compare = exclusive_compare if threshold_source == "percentile" else inclusive_compare
    compare(validated_scores, validated_threshold, out=expected_flags)
    if not np.array_equal(validated_flags, expected_flags):
        if threshold_source is None:
            exclusive_compare(validated_scores, validated_threshold, out=expected_flags)
        if threshold_source is not None or not np.array_equal(validated_flags, expected_flags):
            rule = " and threshold source" if threshold_source is not None else ""
            raise ValueError(
                f"response-time archive flags are inconsistent with its threshold, direction{rule}"
            )

    return (
        validated_scores,
        validated_flags,
        validated_metric,
        validated_direction,
        validated_threshold,
    )


def _read_response_time_archive(archive: NpzFile) -> ResponseTimeArchive:
    """Validate one open NPZ archive and extract its response-time payload."""
    if len(archive.files) != len(set(archive.files)):
        raise ValueError("response-time archive member names must be unique")

    base_members = {
        "schema_version",
        "result_type",
        "n_respondents",
        "metric",
        "flag_direction",
        "threshold",
        "scores",
        "flags",
        "respondent_ids",
    }
    schema_version = _integer_scalar(archive, "schema_version")
    if schema_version == _ARCHIVE_SCHEMA_VERSION:
        allowed_members = base_members
        threshold_source = None
        percentile = None
    elif schema_version == _RESPONSE_TIME_ARCHIVE_SCHEMA_VERSION:
        allowed_members = base_members | {"threshold_source", "percentile"}
        threshold_source = _validate_response_time_threshold_source(
            _string_scalar(archive, "threshold_source")
        )
        percentile = _validate_response_time_percentile(
            _optional_percentile_scalar(archive, "percentile"),
            threshold_source,
        )
    else:
        raise ValueError(
            f"unsupported response-time archive schema version: {schema_version}; "
            f"expected {_ARCHIVE_SCHEMA_VERSION} or "
            f"{_RESPONSE_TIME_ARCHIVE_SCHEMA_VERSION}"
        )
    unexpected = set(archive.files) - allowed_members
    if unexpected:
        raise ValueError(f"response-time archive contains unexpected member: {min(unexpected)}")

    result_type = _string_scalar(archive, "result_type")
    if result_type != "response_time":
        raise ValueError("response-time archive result_type must be 'response_time'")

    n_respondents = _integer_scalar(archive, "n_respondents")
    if n_respondents < 1:
        raise ValueError("response-time archive n_respondents must be positive")

    scores, flags, metric, direction, threshold = _validate_response_time_values(
        _require_member(archive, "scores"),
        _require_member(archive, "flags"),
        _string_scalar(archive, "metric"),
        _string_scalar(archive, "flag_direction"),
        _numeric_scalar(archive, "threshold"),
        expected_respondents=n_respondents,
        threshold_source=threshold_source,
        percentile=percentile,
    )

    respondent_ids = _load_respondent_ids(archive, n_respondents)
    return {
        "schema_version": schema_version,
        "result_type": "response_time",
        "n_respondents": n_respondents,
        "metric": metric,
        "flag_direction": direction,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "percentile": percentile,
        "scores": scores,
        "flags": flags,
        "respondent_ids": respondent_ids,
    }


def _read_response_time_mixture_model(archive: NpzFile) -> ResponseTimeMixtureModel:
    """Validate one open NPZ archive and reconstruct its mixture calibration."""
    if len(archive.files) != len(set(archive.files)):
        raise ValueError("response-time mixture model member names must be unique")

    allowed_members = {
        "schema_version",
        "result_type",
        "n_components",
        "weights",
        "means",
        "variances",
        "log_transform",
    }
    unexpected = set(archive.files) - allowed_members
    if unexpected:
        raise ValueError(
            f"response-time mixture model contains unexpected member: {min(unexpected)}"
        )

    schema_version = _integer_scalar(archive, "schema_version")
    if schema_version != _RESPONSE_TIME_MIXTURE_MODEL_SCHEMA_VERSION:
        raise ValueError(
            "unsupported response-time mixture model schema version: "
            f"{schema_version}; expected {_RESPONSE_TIME_MIXTURE_MODEL_SCHEMA_VERSION}"
        )
    result_type = _string_scalar(archive, "result_type")
    if result_type != "response_time_mixture_model":
        raise ValueError(
            "response-time mixture model result_type must be 'response_time_mixture_model'"
        )
    n_components = _integer_scalar(archive, "n_components")
    if n_components < 2:
        raise ValueError("response-time mixture model n_components must be at least 2")

    model = ResponseTimeMixtureModel(
        weights=_numeric_vector(archive, "weights"),
        means=_numeric_vector(archive, "means"),
        variances=_numeric_vector(archive, "variances"),
        log_transform=_boolean_scalar(archive, "log_transform"),
    )
    if model.n_components != n_components:
        raise ValueError("response-time mixture model parameter lengths must match n_components")
    return model


def save_response_time_mixture_model(
    path: str | Path,
    model: ResponseTimeMixtureModel,
) -> None:
    """Save a mixture calibration as a versioned, pickle-free NPZ archive.

    Validation and parameter copying finish before the destination is opened.
    The archive is streamed into a same-directory temporary path and atomically
    replaces the destination only after all members are written.

    Parameters:
    - path: Explicit destination ending in ``.npz``.
    - model: Calibration returned by ``fit_response_time_mixture()``.
    """
    destination = Path(path)
    if destination.suffix.casefold() != ".npz":
        raise ValueError("response-time mixture model output path must end in .npz")
    if not isinstance(model, ResponseTimeMixtureModel):
        raise TypeError("model must be a ResponseTimeMixtureModel")

    snapshot = ResponseTimeMixtureModel(
        weights=model.weights,
        means=model.means,
        variances=model.variances,
        log_transform=model.log_transform,
    )
    payload = {
        "schema_version": np.asarray(
            _RESPONSE_TIME_MIXTURE_MODEL_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "result_type": np.asarray("response_time_mixture_model", dtype=np.str_),
        "n_components": np.asarray(snapshot.n_components, dtype=np.int64),
        "weights": snapshot.weights,
        "means": snapshot.means,
        "variances": snapshot.variances,
        "log_transform": np.asarray(snapshot.log_transform, dtype=np.bool_),
    }
    _write_npz_archive(destination, payload)


def load_response_time_mixture_model(path: str | Path) -> ResponseTimeMixtureModel:
    """Load and validate a versioned, pickle-free mixture calibration archive.

    The returned model owns read-only parameter copies and can be passed directly
    to ``response_time_mixture_scores()`` for fixed-calibration scoring.

    Parameters:
    - path: Path to a model archive written by
      ``save_response_time_mixture_model()``.
    """
    with _open_npz_archive(path, label="response-time mixture model") as archive:
        return _read_response_time_mixture_model(archive)


def _read_psychsyn_model(archive: NpzFile) -> PsychsynModel:
    """Validate one open NPZ archive and reconstruct its pair calibration."""
    if len(archive.files) != len(set(archive.files)):
        raise ValueError("psychometric pair model member names must be unique")

    allowed_members = {
        "schema_version",
        "result_type",
        "n_items",
        "critval",
        "anto",
        "item_pairs",
    }
    unexpected = set(archive.files) - allowed_members
    if unexpected:
        raise ValueError(f"psychometric pair model contains unexpected member: {min(unexpected)}")

    schema_version = _integer_scalar(archive, "schema_version")
    if schema_version != _PSYCHSYN_MODEL_SCHEMA_VERSION:
        raise ValueError(
            "unsupported psychometric pair model schema version: "
            f"{schema_version}; expected {_PSYCHSYN_MODEL_SCHEMA_VERSION}"
        )
    result_type = _string_scalar(archive, "result_type")
    if result_type != "psychsyn_model":
        raise ValueError("psychometric pair model result_type must be 'psychsyn_model'")

    item_pairs = _require_member(archive, "item_pairs")
    if item_pairs.ndim != 2 or item_pairs.shape[1] != 2 or item_pairs.dtype.kind not in "iu":
        raise ValueError("NPZ archive member item_pairs must be a two-column integer array")
    return PsychsynModel(
        item_pairs=item_pairs,
        n_items=_integer_scalar(archive, "n_items"),
        critval=_numeric_scalar(archive, "critval"),
        anto=_boolean_scalar(archive, "anto"),
    )


def save_psychsyn_model(path: str | Path, model: PsychsynModel) -> None:
    """Save a psychometric pair calibration as a versioned NPZ archive.

    Validation and pair copying finish before the destination is opened. The
    pickle-free archive is streamed to a same-directory temporary path and
    atomically replaces the destination only after all members are written.

    Parameters:
    - path: Explicit destination ending in ``.npz``.
    - model: Calibration returned by ``fit_psychsyn_model()``.
    """
    destination = Path(path)
    if destination.suffix.casefold() != ".npz":
        raise ValueError("psychometric pair model output path must end in .npz")
    if not isinstance(model, PsychsynModel):
        raise TypeError("model must be a PsychsynModel")

    snapshot = PsychsynModel(
        item_pairs=model.item_pairs,
        n_items=model.n_items,
        critval=model.critval,
        anto=model.anto,
    )
    payload = {
        "schema_version": np.asarray(_PSYCHSYN_MODEL_SCHEMA_VERSION, dtype=np.int64),
        "result_type": np.asarray("psychsyn_model", dtype=np.str_),
        "n_items": np.asarray(snapshot.n_items, dtype=np.int64),
        "critval": np.asarray(snapshot.critval, dtype=np.float64),
        "anto": np.asarray(snapshot.anto, dtype=np.bool_),
        "item_pairs": snapshot.item_pairs,
    }
    _write_npz_archive(destination, payload)


def load_psychsyn_model(path: str | Path) -> PsychsynModel:
    """Load and validate a versioned, pickle-free pair calibration archive.

    The returned model owns a read-only pair array and can be passed directly to
    ``psychsyn_model_scores()`` for fixed-calibration scoring.

    Parameters:
    - path: Path to an archive written by ``save_psychsyn_model()``.
    """
    with _open_npz_archive(path, label="psychometric pair model") as archive:
        return _read_psychsyn_model(archive)


def save_score_archive(
    path: str | Path,
    scores: Mapping[str, ArrayLike],
    *,
    result_type: ScoreArchiveResultType = "screen",
    respondent_ids: Sequence[str] | None = None,
    errors: Mapping[str, str] | None = None,
) -> None:
    """
    Save reusable registered-index scores as a versioned, pickle-free NPZ archive.

    Score and metadata validation completes before the destination is opened.
    Compatible float64 arrays are streamed without an intermediate score matrix,
    and mapping insertion order is preserved. Composite archives accept only
    indices supported by ``composite_scores()``.

    Parameters:
    - path: Explicit destination ending in ``.npz``.
    - scores: Ordered mapping of registered index names to aligned score vectors.
    - result_type: ``"screen"`` or ``"composite"``.
    - respondent_ids: Optional aligned, unique, nonblank string identifiers.
    - errors: Optional ordered mapping of failed index names to nonblank messages.

    Example:
        >>> from ier import load_score_archive, save_score_archive, screen_scores
        >>> scores = {"irv": [0.1, 0.7], "longstring": [3.0, 8.0]}
        >>> save_score_archive("scores.npz", scores, respondent_ids=["a", "b"])
        >>> saved = load_score_archive("scores.npz")
        >>> updated = screen_scores(saved["scores"], percentile=95)
    """
    destination = Path(path)
    if destination.suffix.casefold() != ".npz":
        raise ValueError("score archive output path must end in .npz")

    validated_result_type = _validate_result_type(result_type)
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be a mapping of registered index names to score arrays")
    score_items = list(scores.items())
    score_names = [name for name, _ in score_items]
    _validate_archive_index_names(score_names, validated_result_type)
    validated_scores, n_respondents = validate_score_vectors(dict(score_items))

    if errors is None:
        validated_errors: dict[str, str] = {}
    else:
        if not isinstance(errors, Mapping):
            raise TypeError("errors must be a mapping of registered index names to messages")
        error_items = list(errors.items())
        validated_errors = _validate_error_metadata(
            [name for name, _ in error_items],
            [message for _, message in error_items],
            set(score_names),
            validated_result_type,
        )

    validated_ids: list[str] | None = None
    if respondent_ids is not None:
        if isinstance(respondent_ids, (str, bytes)):
            raise TypeError("respondent_ids must be a sequence of strings")
        validated_ids = _validate_respondent_ids(list(respondent_ids), n_respondents)

    payload = {
        "schema_version": np.asarray(_ARCHIVE_SCHEMA_VERSION, dtype=np.int64),
        "result_type": np.asarray(validated_result_type, dtype=np.str_),
        "n_respondents": np.asarray(n_respondents, dtype=np.int64),
        "index_names": np.asarray(score_names, dtype=np.str_),
        "error_names": np.asarray(list(validated_errors), dtype=np.str_),
        "error_messages": np.asarray(list(validated_errors.values()), dtype=np.str_),
    }
    for name, values in validated_scores.items():
        payload[f"score__{name}"] = values
    if validated_ids is not None:
        payload["respondent_ids"] = np.asarray(validated_ids, dtype=np.str_)
    _write_npz_archive(destination, payload)


def save_response_time_archive(
    path: str | Path,
    scores: ArrayLike,
    flags: ArrayLike,
    *,
    threshold: float,
    metric: ResponseTimeMetric = "median",
    flag_direction: ResponseTimeFlagDirection = "low",
    respondent_ids: Sequence[str] | None = None,
    threshold_source: ResponseTimeThresholdSource | None = None,
    percentile: float | None = None,
) -> None:
    """
    Save reusable response-time results as a versioned, pickle-free NPZ archive.

    Scores, Boolean flags, metric/direction compatibility, the finite threshold,
    and optional respondent identifiers are validated before the destination is
    opened. With no cutoff provenance, the writer preserves the legacy v1
    schema and accepts either flag rule. Providing ``threshold_source`` or
    ``percentile`` writes schema v2 and validates the exact fixed-inclusive or
    percentile-exclusive rule.

    Parameters:
    - path: Explicit destination ending in ``.npz``.
    - scores: Per-respondent direct timing scores or mixture probabilities.
    - flags: Aligned Boolean decisions produced from the recorded threshold.
    - threshold: Resolved finite cutoff in the score's units.
    - metric: Timing metric represented by the score vector.
    - flag_direction: Suspicious tail, ``"low"`` or ``"high"``.
    - respondent_ids: Optional aligned, unique, nonblank string identifiers.
    - threshold_source: Optional ``"fixed"`` or ``"percentile"`` provenance.
    - percentile: Requested percentile when the cutoff is percentile-derived.

    Example:
        >>> from ier import response_time_score_flags, save_response_time_archive
        >>> scores = [0.5, 1.2, 2.0]
        >>> flags = response_time_score_flags(scores, threshold=1.0)
        >>> save_response_time_archive(
        ...     "timing.npz", scores, flags, threshold=1.0, metric="median"
        ... )
    """
    destination = Path(path)
    if destination.suffix.casefold() != ".npz":
        raise ValueError("response-time archive output path must end in .npz")

    has_provenance = threshold_source is not None or percentile is not None
    validated_source: ResponseTimeThresholdSource | None = None
    if has_provenance:
        validated_source = _validate_response_time_threshold_source(
            "percentile" if threshold_source is None else threshold_source
        )
        percentile = _validate_response_time_percentile(percentile, validated_source)

    validated_scores, validated_flags, validated_metric, validated_direction, cutoff = (
        _validate_response_time_values(
            scores,
            flags,
            metric,
            flag_direction,
            threshold,
            threshold_source=validated_source,
            percentile=percentile,
        )
    )
    validated_ids: list[str] | None = None
    if respondent_ids is not None:
        if isinstance(respondent_ids, (str, bytes)):
            raise TypeError("respondent_ids must be a sequence of strings")
        validated_ids = _validate_respondent_ids(
            list(respondent_ids),
            len(validated_scores),
        )

    payload = {
        "schema_version": np.asarray(
            _RESPONSE_TIME_ARCHIVE_SCHEMA_VERSION if has_provenance else _ARCHIVE_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "result_type": np.asarray("response_time", dtype=np.str_),
        "n_respondents": np.asarray(len(validated_scores), dtype=np.int64),
        "metric": np.asarray(validated_metric, dtype=np.str_),
        "flag_direction": np.asarray(validated_direction, dtype=np.str_),
        "threshold": np.asarray(cutoff, dtype=np.float64),
        "scores": validated_scores,
        "flags": validated_flags,
    }
    if validated_source is not None:
        payload["threshold_source"] = np.asarray(validated_source, dtype=np.str_)
        payload["percentile"] = np.asarray(
            np.nan if percentile is None else percentile,
            dtype=np.float64,
        )
    if validated_ids is not None:
        payload["respondent_ids"] = np.asarray(validated_ids, dtype=np.str_)
    _write_npz_archive(destination, payload)


def load_score_archive(path: str | Path) -> ScoreArchive:
    """
    Load reusable registered-index scores from a versioned NPZ archive.

    The loader always disables pickling and validates schema version, result type,
    member names, registry membership, vector shape, respondent alignment,
    optional identifiers, and soft-failure metadata. Screen archives are reusable
    directly. Full composite CLI archives must have been written with
    ``--include-components`` so their raw public index scores are present;
    compact archives from ``save_score_archive()`` are directly compatible.

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
    with _open_npz_archive(path, label="score") as archive:
        return _read_score_archive(archive)


def merge_score_archives(
    paths: Sequence[str | Path],
    *,
    result_type: ScoreArchiveResultType = "screen",
) -> ScoreArchive:
    """
    Load and merge independently computed registered-index score archives.

    At least two validated score archives are required. Index order follows the
    input archive order and duplicate score indices are rejected. All archives
    must describe the same respondents: when every archive contains identifiers,
    score vectors are aligned to the first archive's identifier order; when none
    contains identifiers, equal respondent counts are treated as matching row
    order. Mixing identified and unidentified archives is rejected.

    A successful score supersedes soft-failure metadata for the same index.
    Repeated identical failures are retained once, while conflicting messages
    are rejected. Aligned arrays are reused without copying; only vectors whose
    respondent order differs are reordered.

    Parameters:
    - paths: Ordered sequence of score archive paths to merge.
    - result_type: Output validation contract, ``"screen"`` or ``"composite"``.

    Returns:
    - A validated in-memory ``ScoreArchive`` ready for ``screen_scores()``,
      ``composite_scores()``, or ``save_score_archive()``.

    Example:
        >>> from ier import merge_score_archives, save_score_archive
        >>> merged = merge_score_archives(["patterns.npz", "consistency.npz"])
        >>> save_score_archive(
        ...     "combined.npz",
        ...     merged["scores"],
        ...     result_type=merged["result_type"],
        ...     respondent_ids=merged["respondent_ids"],
        ...     errors=merged["errors"],
        ... )
    """
    if isinstance(paths, (str, bytes, Path)):
        raise TypeError("paths must be a sequence of score archive paths")
    archive_paths = list(paths)
    if len(archive_paths) < 2:
        raise ValueError("at least two score archives are required to merge")

    validated_result_type = _validate_result_type(result_type)
    archives = [load_score_archive(path) for path in archive_paths]
    n_respondents = archives[0]["n_respondents"]
    if any(archive["n_respondents"] != n_respondents for archive in archives[1:]):
        raise ValueError("score archives must contain the same number of respondents")

    identified = [archive["respondent_ids"] is not None for archive in archives]
    if any(identified) and not all(identified):
        raise ValueError("score archives must all include respondent IDs or all omit them")

    canonical_ids = archives[0]["respondent_ids"]
    row_orders: list[np.ndarray | None] = [None]
    if canonical_ids is None:
        row_orders.extend(None for _ in archives[1:])
    else:
        canonical_set = set(canonical_ids)
        for archive in archives[1:]:
            respondent_ids = archive["respondent_ids"]
            assert respondent_ids is not None
            if set(respondent_ids) != canonical_set:
                raise ValueError("score archive respondent ID sets must match")
            if respondent_ids == canonical_ids:
                row_orders.append(None)
                continue
            positions = {respondent_id: index for index, respondent_id in enumerate(respondent_ids)}
            row_orders.append(
                np.fromiter(
                    (positions[respondent_id] for respondent_id in canonical_ids),
                    dtype=np.intp,
                    count=n_respondents,
                )
            )

    merged_scores: dict[str, np.ndarray] = {}
    for archive, row_order in zip(archives, row_orders, strict=True):
        for name, values in archive["scores"].items():
            if name in merged_scores:
                raise ValueError(f"duplicate score index across archives: {name}")
            merged_scores[name] = values if row_order is None else values[row_order]
    _validate_archive_index_names(list(merged_scores), validated_result_type)

    merged_errors: dict[str, str] = {}
    for archive in archives:
        for name, message in archive["errors"].items():
            if name in merged_scores:
                continue
            previous = merged_errors.get(name)
            if previous is not None and previous != message:
                raise ValueError(f"conflicting error messages across archives for index: {name}")
            merged_errors.setdefault(name, message)
    validated_errors = _validate_error_metadata(
        list(merged_errors),
        list(merged_errors.values()),
        set(merged_scores),
        validated_result_type,
    )

    return {
        "schema_version": _ARCHIVE_SCHEMA_VERSION,
        "result_type": validated_result_type,
        "n_respondents": n_respondents,
        "scores": merged_scores,
        "respondent_ids": None if canonical_ids is None else canonical_ids.copy(),
        "errors": validated_errors,
    }


def load_archive(path: str | Path) -> InspectableArchive:
    """
    Load and auto-detect any supported result or model archive.

    Pickling is always disabled. The archive's declared ``result_type`` selects
    the complete score, response-time, or model validator, so callers can
    inspect supported archives without opening raw NPZ members or guessing which
    specialized loader to call. Generic model results retain validated read-only
    parameter arrays; the dedicated model loader returns the scoring dataclass.

    Parameters:
    - path: Path to a supported versioned NPZ result archive.

    Returns:
    - A fully validated score, response-time, or model archive mapping.

    Example:
        >>> from ier import load_archive
        >>> saved = load_archive("results.npz")
        >>> print(saved["result_type"])
    """
    with _open_npz_archive(path, label="result") as archive:
        result_type = _string_scalar(archive, "result_type")
        if result_type == "response_time":
            return _read_response_time_archive(archive)
        if result_type in {"screen", "composite"}:
            return _read_score_archive(archive)
        if result_type == "response_time_mixture_model":
            mixture_model = _read_response_time_mixture_model(archive)
            return {
                "schema_version": _RESPONSE_TIME_MIXTURE_MODEL_SCHEMA_VERSION,
                "result_type": "response_time_mixture_model",
                "n_components": mixture_model.n_components,
                "fast_component": mixture_model.fast_component,
                "log_transform": mixture_model.log_transform,
                "weights": mixture_model.weights,
                "means": mixture_model.means,
                "variances": mixture_model.variances,
            }
        if result_type == "psychsyn_model":
            pair_model = _read_psychsyn_model(archive)
            return {
                "schema_version": _PSYCHSYN_MODEL_SCHEMA_VERSION,
                "result_type": "psychsyn_model",
                "n_items": pair_model.n_items,
                "n_pairs": pair_model.n_pairs,
                "critval": pair_model.critval,
                "anto": pair_model.anto,
                "item_pairs": pair_model.item_pairs,
            }
        raise ValueError(
            "archive result_type must be 'screen', 'composite', 'response_time', "
            "'response_time_mixture_model', or 'psychsyn_model'"
        )


def load_response_time_archive(path: str | Path) -> ResponseTimeArchive:
    """
    Load response-time results from a versioned, pickle-free NPZ archive.

    The loader validates every schema field, the metric and suspicious-tail
    pairing, vector shape and respondent alignment, optional identifiers, and
    agreement between the stored flags and threshold. The returned score vector
    can be passed directly to ``response_time_score_flags()`` to apply a new
    fixed or percentile cutoff without recomputing the timing metric.

    Parameters:
    - path: Path to a response-time NPZ archive written by the CLI.

    Returns:
    - A ``ResponseTimeArchive`` containing scores, flags, cutoff metadata, and
      optional respondent identifiers.

    Example:
        >>> from ier import load_response_time_archive, response_time_score_flags
        >>> saved = load_response_time_archive("timing.npz")
        >>> strict = response_time_score_flags(
        ...     saved["scores"], cutoff_percentile=1,
        ...     direction=saved["flag_direction"],
        ... )
    """
    with _open_npz_archive(path, label="response-time") as archive:
        return _read_response_time_archive(archive)
