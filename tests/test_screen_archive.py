"""Complete screen() results persist with validated, recomputed decisions."""

from __future__ import annotations

import math
import tracemalloc
from typing import TYPE_CHECKING, Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pytest

from ier import (
    load_score_archive,
    load_screen_archive,
    save_score_archive,
    save_screen_archive,
    screen,
    screen_scores,
)
from ier.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ier import ScreenResult

SCREEN_MEMBERS = [
    "schema_version",
    "result_type",
    "n_respondents",
    "n_indices",
    "min_flags",
    "index_names",
    "thresholds",
    "threshold_sources",
    "percentiles",
    "flag_counts",
    "valid_index_counts",
    "consensus_eligible",
    "consensus_flags",
    "summary_columns",
    "summary_statistics",
    "summary_n_flagged",
    "summary_n_valid",
    "summary_n_unavailable",
    "summary_flag_rate",
    "min_valid_indices",
    "error_names",
    "error_messages",
    "score__irv",
    "flag__irv",
    "score__longstring",
    "flag__longstring",
    "score__onset",
    "flag__onset",
    "score__person_total",
    "flag__person_total",
    "respondent_ids",
]


def _responses() -> np.ndarray:
    rng = np.random.default_rng(11)
    data = rng.integers(1, 6, size=(80, 30)).astype(float)
    # Careless onset: attentive answers followed by straightlining.
    data[::9, 15:] = 3.0
    data[::13, rng.integers(0, 30)] = np.nan
    data[5] = np.nan
    return data


def _mixed_result() -> ScreenResult:
    """Fixed, percentile, presence, and failed indices with a completeness rule."""
    return screen(
        _responses(),
        indices=["irv", "longstring", "onset", "mad", "person_total"],
        thresholds={"irv": 0.9},
        percentiles={"longstring": 90.0},
        min_flags=2,
        min_valid_indices=2,
    )


def _assert_same_result(actual: ScreenResult, expected: ScreenResult) -> None:
    assert list(actual) == list(expected)
    for key in (
        "thresholds",
        "threshold_sources",
        "percentiles",
        "min_flags",
        "min_valid_indices",
        "n_indices",
        "indices_used",
        "errors",
        "n_respondents",
    ):
        assert actual[key] == expected[key], key  # type: ignore[literal-required]
    for key in ("scores", "flags"):
        assert list(actual[key]) == list(expected[key])  # type: ignore[literal-required]
        for name, values in expected[key].items():  # type: ignore[literal-required]
            loaded = actual[key][name]  # type: ignore[literal-required]
            np.testing.assert_array_equal(loaded, values, err_msg=f"{key}:{name}")
            assert loaded.dtype == values.dtype
    for key in ("flag_counts", "valid_index_counts", "consensus_eligible", "consensus_flags"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)  # type: ignore[literal-required]
        assert actual[key].dtype == expected[key].dtype  # type: ignore[literal-required]
    assert list(actual["summary"]) == list(expected["summary"])
    for name, statistics in expected["summary"].items():
        assert list(actual["summary"][name]) == list(statistics)
        for field, value in statistics.items():
            loaded_value = actual["summary"][name][field]  # type: ignore[literal-required]
            assert type(loaded_value) is type(value), (name, field)
            assert loaded_value == value or (math.isnan(loaded_value) and math.isnan(value))


def _archive_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("respondent_ids", [None, [f"répondant-{index}" for index in range(80)]])
def test_complete_screen_result_round_trips(
    tmp_path: Path, compressed: bool, respondent_ids: list[str] | None
) -> None:
    original = _mixed_result()
    assert original["errors"].keys() == {"mad"}
    assert set(original["threshold_sources"].values()) == {"fixed", "percentile", "presence"}
    path = tmp_path / "screening.npz"

    save_screen_archive(path, original, respondent_ids=respondent_ids, compressed=compressed)

    with ZipFile(path) as container:
        assert {member.compress_type for member in container.infolist()} == {
            ZIP_DEFLATED if compressed else ZIP_STORED
        }
    saved = load_screen_archive(path)
    assert list(saved) == ["schema_version", "result", "respondent_ids"]
    assert saved["schema_version"] == 1
    assert saved["respondent_ids"] == respondent_ids
    _assert_same_result(saved["result"], original)
    # Score-only reuse of the same archive is unchanged.
    scores = load_score_archive(path)
    assert scores["errors"] == original["errors"]
    for name, values in original["scores"].items():
        np.testing.assert_array_equal(scores["scores"][name], values)


@pytest.mark.parametrize("compressed", [False, True])
def test_percentile_ties_survive_round_trip_unlike_fixed_threshold_replay(
    tmp_path: Path, compressed: bool
) -> None:
    rng = np.random.default_rng(5)
    scores = {
        "longstring": rng.integers(1, 10, size=1000).astype(float),
        "irv": rng.normal(size=1000),
    }
    original = screen_scores(scores, percentile=90)
    path = tmp_path / "ties.npz"
    save_screen_archive(path, original, compressed=compressed)

    restored = load_screen_archive(path)["result"]

    _assert_same_result(restored, original)
    assert restored["threshold_sources"]["longstring"] == "percentile"
    assert not restored["flags"]["longstring"].any()
    # Replaying recorded cutoffs as explicit thresholds makes them inclusive.
    thresholds = {name: float(restored["thresholds"][name] or 0.0) for name in scores}
    replayed = screen_scores(restored["scores"], thresholds=thresholds)
    assert replayed["threshold_sources"]["longstring"] == "fixed"
    assert replayed["flags"]["longstring"].sum() > 100
    np.testing.assert_array_equal(replayed["flags"]["irv"], restored["flags"]["irv"])


def test_all_failed_screen_result_round_trips(tmp_path: Path) -> None:
    original = screen(
        [[1.0, 2.0, 3.0], [3.0, 3.0, 3.0]],
        indices=["mad", "evenodd"],
        min_flags=1,
        min_valid_indices=2,
    )
    assert original["scores"] == {}
    path = tmp_path / "failed.npz"
    save_screen_archive(path, original, respondent_ids=["A", "B"])

    saved = load_screen_archive(path)

    assert saved["respondent_ids"] == ["A", "B"]
    _assert_same_result(saved["result"], original)


def test_cli_screen_archive_loads_with_unchanged_members(tmp_path: Path) -> None:
    data = _responses()
    source = tmp_path / "responses.csv"
    rows = ["id," + ",".join(f"q{column}" for column in range(data.shape[1]))]
    for index, row in enumerate(data):
        cells = ["" if math.isnan(value) else str(int(value)) for value in row]
        rows.append(f"case-{index}," + ",".join(cells))
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    cli_path = tmp_path / "cli.npz"
    assert (
        main(
            [
                "screen",
                str(source),
                "--id-column",
                "id",
                "--indices",
                "irv",
                "longstring",
                "onset",
                "mad",
                "person_total",
                "--threshold",
                "irv=0.9",
                "--index-percentile",
                "longstring=90",
                "--min-flags",
                "2",
                "--min-valid-indices",
                "2",
                "--format",
                "npz",
                "--output",
                str(cli_path),
            ]
        )
        == 0
    )
    ids = [f"case-{index}" for index in range(len(data))]
    public_path = tmp_path / "public.npz"
    save_screen_archive(public_path, _mixed_result(), respondent_ids=ids)

    saved = load_screen_archive(cli_path)

    assert saved["respondent_ids"] == ids
    _assert_same_result(saved["result"], _mixed_result())
    cli_payload = _archive_payload(cli_path)
    public_payload = _archive_payload(public_path)
    assert list(cli_payload) == SCREEN_MEMBERS
    assert list(public_payload) == SCREEN_MEMBERS
    for name, values in cli_payload.items():
        np.testing.assert_array_equal(public_payload[name], values, err_msg=name)
        assert public_payload[name].dtype == values.dtype


def test_restored_result_is_plottable_without_recomputing(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ier import plot_distributions, plot_flag_counts, plot_flagged_heatmap

    path = tmp_path / "screening.npz"
    save_screen_archive(path, _mixed_result())
    restored = load_screen_archive(path)["result"]
    for plot in (plot_distributions, plot_flag_counts, plot_flagged_heatmap):
        figure = plot(restored)  # type: ignore[operator]
        assert isinstance(figure, plt.Figure)
        plt.close(figure)


def _flip_first(payload: dict[str, np.ndarray], name: str) -> None:
    payload[name] = payload[name].copy()
    payload[name][0] = not payload[name][0]


def _set_position(name: str, position: int, value: Any) -> Callable[[dict[str, np.ndarray]], None]:
    def update(payload: dict[str, np.ndarray]) -> None:
        changed = payload[name].copy()
        changed[position] = value
        payload[name] = changed

    return update


def _replace(name: str, value: Any) -> Callable[[dict[str, np.ndarray]], None]:
    def update(payload: dict[str, np.ndarray]) -> None:
        payload[name] = np.asarray(value)

    return update


def _remove(name: str) -> Callable[[dict[str, np.ndarray]], None]:
    def update(payload: dict[str, np.ndarray]) -> None:
        del payload[name]

    return update


# Index order in the mixed archive: irv (fixed), longstring (percentile),
# onset (presence), person_total (percentile); mad is a retained failure.
CORRUPTIONS: list[tuple[str, Callable[[dict[str, np.ndarray]], None], str]] = [
    ("irv flag", lambda p: _flip_first(p, "flag__irv"), "flags for irv are inconsistent"),
    (
        "percentile flag",
        lambda p: _flip_first(p, "flag__longstring"),
        "flags for longstring are inconsistent with its scores and percentile rule",
    ),
    (
        "presence flag",
        lambda p: _flip_first(p, "flag__onset"),
        "flags for onset are inconsistent with its scores and presence rule",
    ),
    ("flag dtype", _replace("flag__irv", np.zeros(80, dtype=np.int64)), "boolean vector"),
    ("flag length", _replace("flag__irv", np.zeros(79, dtype=bool)), "match n_respondents"),
    ("flag count", _set_position("flag_counts", 0, 4), "flag_counts are inconsistent"),
    ("flag count dtype", _replace("flag_counts", np.zeros(80)), "an integer vector"),
    (
        "valid count",
        _set_position("valid_index_counts", 0, 0),
        "valid_index_counts are inconsistent",
    ),
    ("eligibility", lambda p: _flip_first(p, "consensus_eligible"), "consensus_eligible is"),
    ("consensus", lambda p: _flip_first(p, "consensus_flags"), "consensus_flags are"),
    (
        "fixed source",
        _set_position("threshold_sources", 1, "fixed"),
        "percentile for fixed-threshold longstring must be NaN",
    ),
    (
        "presence source",
        _set_position("threshold_sources", 0, "presence"),
        "threshold source for irv must be 'fixed' or 'percentile'",
    ),
    (
        "registry presence",
        _set_position("threshold_sources", 2, "fixed"),
        "threshold source for onset must be 'presence'",
    ),
    (
        "unknown source",
        _set_position("threshold_sources", 0, "guess"),
        "must be 'fixed', 'percentile', or 'presence'",
    ),
    ("source length", _replace("threshold_sources", ["fixed"]), "must match index_names"),
    (
        "source type",
        _replace("threshold_sources", [1, 2, 3, 4]),
        "member threshold_sources must be a Unicode string vector",
    ),
    ("result type", _replace("result_type", "composite"), "result_type must be 'screen'"),
    ("schema", _replace("schema_version", 2), "unsupported screen archive schema version"),
    ("respondents", _replace("n_respondents", 0), "screen archive n_respondents must be positive"),
    (
        "presence threshold",
        _set_position("thresholds", 2, 1.0),
        "threshold for presence-flagged onset must be NaN",
    ),
    ("missing threshold", _set_position("thresholds", 0, np.nan), "irv must be finite"),
    ("threshold dtype", _replace("thresholds", np.zeros(4, dtype=np.int64)), "floating-point"),
    ("threshold length", _replace("thresholds", [0.5]), "thresholds must match index_names"),
    (
        "presence percentile",
        _set_position("percentiles", 2, 95.0),
        "percentile for presence-flagged onset must be NaN",
    ),
    (
        "percentile range",
        _set_position("percentiles", 1, 150.0),
        "percentile for longstring must be between 0 and 100",
    ),
    ("percentile missing", _set_position("percentiles", 1, np.nan), "between 0 and 100"),
    ("index count", _replace("n_indices", 5), "n_indices must match index_names"),
    ("min flags", _replace("min_flags", 0), "min_flags must be positive"),
    ("min flags type", _replace("min_flags", [1]), "min_flags must be an integer scalar"),
    (
        "completeness",
        _replace("min_valid_indices", 6),
        r"min_valid_indices must be between 1 and the number of selected indices \(5\)",
    ),
    ("summary columns", _replace("summary_columns", ["std", "mean", "min", "max"]), "columns"),
    (
        "summary matrix",
        _replace("summary_statistics", np.zeros((4, 3))),
        "summary_statistics must be a floating-point matrix",
    ),
    (
        "summary flagged",
        _set_position("summary_n_flagged", 0, 79),
        "summary counts for irv are inconsistent",
    ),
    (
        "summary valid",
        _set_position("summary_n_valid", 3, 0),
        "summary counts for person_total are inconsistent",
    ),
    (
        "summary unavailable",
        _set_position("summary_n_unavailable", 2, 81),
        "summary counts for onset are inconsistent",
    ),
    ("summary rate", _replace("summary_flag_rate", [0.1]), "summary_flag_rate must match"),
    ("undeclared flag", _replace("flag__mad", np.zeros(80, dtype=bool)), "unexpected member"),
    ("unexpected", _replace("extra", 1), "screen archive contains unexpected member: extra"),
    ("missing flag", _remove("flag__onset"), "missing required member: flag__onset"),
    ("missing consensus", _remove("consensus_flags"), "missing required member: consensus_flags"),
    ("duplicate IDs", _replace("respondent_ids", ["case"] * 80), "IDs must be unique"),
]


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [pytest.param(corrupt, message, id=label) for label, corrupt, message in CORRUPTIONS],
)
def test_tampered_screen_archive_is_rejected(
    tmp_path: Path,
    corrupt: Callable[[dict[str, np.ndarray]], None],
    message: str,
) -> None:
    valid = tmp_path / "valid.npz"
    save_screen_archive(valid, _mixed_result(), respondent_ids=[f"id-{i}" for i in range(80)])
    payload = _archive_payload(valid)
    corrupt(payload)
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **payload)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        load_screen_archive(tampered)


def test_score_only_and_empty_archives_have_actionable_errors(tmp_path: Path) -> None:
    score_only = tmp_path / "scores.npz"
    save_score_archive(score_only, {"irv": [0.1, 0.2]})
    with pytest.raises(ValueError, match="use load_score_archive"):
        load_screen_archive(score_only)

    failed = tmp_path / "failed.npz"
    save_screen_archive(failed, screen([[1.0, 2.0], [2.0, 1.0]], indices=["mad"]))
    payload = _archive_payload(failed)
    payload["error_names"] = np.asarray([], dtype=np.str_)
    payload["error_messages"] = np.asarray([], dtype=np.str_)
    empty = tmp_path / "empty.npz"
    np.savez(empty, **payload)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not contain index scores or failures"):
        load_screen_archive(empty)

    timing = tmp_path / "timing.npz"
    timing.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="screen archive could not be read"):
        load_screen_archive(timing)


def _invalid_result(kind: str) -> object:
    """Return the valid mixed result or one deliberately inconsistent variant."""
    original = _mixed_result()
    changes: dict[str, object] = {
        "valid": {},
        "reordered": {"scores": dict(reversed(original["scores"].items()))},
        "flag": {"flags": {**original["flags"], "irv": ~original["flags"]["irv"]}},
        "counts": {"flag_counts": original["flag_counts"] + 1},
        "empty": {"scores": {}, "flags": {}, "indices_used": [], "errors": {}},
        "errors": {"errors": [("mad", "failed")]},
    }
    if kind == "not-mapping":
        return [original]
    return {**original, **changes[kind]}  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("kind", "options", "error", "message"),
    [
        ("not-mapping", {}, TypeError, "ScreenResult mapping"),
        ("reordered", {}, ValueError, "scores must follow indices_used"),
        ("flag", {}, ValueError, "flags for irv are inconsistent"),
        ("counts", {}, ValueError, "flag_counts are inconsistent"),
        ("empty", {}, ValueError, "must contain index scores or failures"),
        ("errors", {}, TypeError, "errors must be a mapping"),
        ("valid", {"respondent_ids": "ab"}, TypeError, "sequence of strings"),
        ("valid", {"respondent_ids": ["a"]}, ValueError, "ID count"),
        (
            "valid",
            {"respondent_ids": ["x\0", *(f"id-{i}" for i in range(79))]},
            ValueError,
            "cannot end with a NUL",
        ),
        ("valid", {"compressed": 1}, ValueError, "compressed must be a boolean"),
        (
            "valid",
            {"compression_level": 9},
            ValueError,
            "compression_level requires compressed=True",
        ),
    ],
)
def test_public_writer_validates_result_before_touching_destination(
    tmp_path: Path,
    kind: str,
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    destination = tmp_path / "existing.npz"
    destination.write_bytes(b"previous complete result")

    with pytest.raises(error, match=message):
        save_screen_archive(destination, _invalid_result(kind), **options)  # type: ignore[arg-type]

    assert destination.read_bytes() == b"previous complete result"
    assert list(tmp_path.iterdir()) == [destination]


def test_public_writer_requires_npz_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="screen archive output path must end in .npz"):
        save_screen_archive(tmp_path / "screening.bin", _mixed_result())
    assert list(tmp_path.iterdir()) == []


def _rewrite(source: Path, target: Path, **members: np.ndarray) -> None:
    """Copy an archive while replacing selected members, as a hand-edited file would."""
    payload = _archive_payload(source)
    payload.update(members)
    np.savez(target, **payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_respondents", [2**45, 20_000_000])
def test_all_failed_archive_header_cannot_size_allocations(
    tmp_path: Path, n_respondents: int
) -> None:
    source = tmp_path / "failed.npz"
    failed = screen_scores({}, errors={"mad": "item pairs were not configured"}, n_respondents=3)
    save_screen_archive(source, failed)
    forged = tmp_path / "forged.npz"
    _rewrite(source, forged, n_respondents=np.asarray(n_respondents, dtype=np.int64))

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="flag_counts must match n_respondents"):
            load_screen_archive(forged)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The 3-element stored vectors are rejected before any header-sized buffer exists.
    assert peak < 8 * 2**20


def test_all_failed_result_with_misaligned_counts_is_rejected_before_writing(
    tmp_path: Path,
) -> None:
    result = screen_scores({}, errors={"mad": "item pairs were not configured"}, n_respondents=3)
    destination = tmp_path / "failed.npz"
    with pytest.raises(ValueError, match="flag_counts must match n_respondents"):
        save_screen_archive(destination, {**result, "n_respondents": 2**40})
    assert not destination.exists()


def _edited_statistics(path: Path, position: int, column: int, value: float) -> np.ndarray:
    statistics = _archive_payload(path)["summary_statistics"].copy()
    statistics[position, column] = value
    return statistics


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("flag_rate", "summary flag rate for longstring is inconsistent with its counts"),
        ("min", "summary min and max for longstring are inconsistent with its scores"),
        ("max", "summary min and max for longstring are inconsistent with its scores"),
    ],
)
def test_summary_rates_and_extrema_must_match_restored_scores(
    tmp_path: Path, field: str, message: str
) -> None:
    valid = tmp_path / "valid.npz"
    save_screen_archive(valid, _mixed_result())
    forged = tmp_path / "forged.npz"
    if field == "flag_rate":
        rates = _archive_payload(valid)["summary_flag_rate"].copy()
        rates[1] = 0.99
        _rewrite(valid, forged, summary_flag_rate=rates)
    else:
        column = 2 if field == "min" else 3
        _rewrite(valid, forged, summary_statistics=_edited_statistics(valid, 1, column, 1e6))

    with pytest.raises(ValueError, match=message):
        load_screen_archive(forged)

    edited = _mixed_result()
    edited["summary"]["longstring"] = {**edited["summary"]["longstring"], field: 0.5}  # type: ignore[misc]
    destination = tmp_path / "edited.npz"
    with pytest.raises(ValueError, match=message):
        save_screen_archive(destination, edited)
    assert not destination.exists()


def test_unavailable_index_summary_must_stay_nan(tmp_path: Path) -> None:
    result = screen_scores({"irv": [0.1, 0.5, 0.9], "markov": [np.nan] * 3}, min_flags=1)
    valid = tmp_path / "valid.npz"
    save_screen_archive(valid, result)
    assert math.isnan(load_screen_archive(valid)["result"]["summary"]["markov"]["flag_rate"])

    rates = _archive_payload(valid)["summary_flag_rate"].copy()
    rates[1] = 0.0
    forged = tmp_path / "forged.npz"
    _rewrite(valid, forged, summary_flag_rate=rates)
    with pytest.raises(ValueError, match="summary flag rate for markov"):
        load_screen_archive(forged)
    _rewrite(valid, forged, summary_statistics=_edited_statistics(valid, 1, 2, 0.0))
    with pytest.raises(ValueError, match="summary min and max for markov"):
        load_screen_archive(forged)


def test_summary_means_and_deviations_are_rebuilt_from_scores(tmp_path: Path) -> None:
    original = _mixed_result()
    valid = tmp_path / "valid.npz"
    save_screen_archive(valid, original)
    statistics = _archive_payload(valid)["summary_statistics"].copy()
    statistics[:, :2] = [1e6, -1.0]
    forged = tmp_path / "forged.npz"
    _rewrite(valid, forged, summary_statistics=statistics)

    _assert_same_result(load_screen_archive(forged)["result"], original)

    # The writer stores the same rebuilt values rather than hand-edited ones.
    edited = _mixed_result()
    edited["summary"]["irv"] = {**edited["summary"]["irv"], "mean": 1e6, "std": -1.0}
    rewritten = tmp_path / "rewritten.npz"
    save_screen_archive(rewritten, edited)
    np.testing.assert_array_equal(
        _archive_payload(rewritten)["summary_statistics"],
        _archive_payload(valid)["summary_statistics"],
    )
    _assert_same_result(load_screen_archive(rewritten)["result"], original)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("min_flags", 2**63),
        ("min_flags", np.uint64(2**63)),
        ("n_indices", 2**64),
        ("min_valid_indices", -(2**63) - 1),
    ],
)
def test_writer_rejects_settings_outside_the_int64_member_range(
    tmp_path: Path, setting: str, value: object
) -> None:
    destination = tmp_path / "screening.npz"
    with pytest.raises(ValueError, match=f"{setting} must fit in a 64-bit signed integer"):
        save_screen_archive(destination, {**_mixed_result(), setting: value})  # type: ignore[misc]
    assert not destination.exists()


def test_largest_int64_min_flags_round_trips(tmp_path: Path) -> None:
    # screen() bounds min_flags only from below, so the int64 limit is a valid result.
    original = screen_scores({"irv": [0.1, 0.5, 0.9]}, min_flags=2**63 - 1)
    path = tmp_path / "screening.npz"
    save_screen_archive(path, original)
    _assert_same_result(load_screen_archive(path)["result"], original)


def test_cli_rejects_min_flags_beyond_int64_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "responses.csv"
    source.write_text("q1,q2,q3\n1,2,3\n3,3,3\n2,4,1\n", encoding="utf-8")
    destination = tmp_path / "screening.npz"

    code = main(
        [
            "screen",
            str(source),
            "--indices",
            "irv",
            "--min-flags",
            str(2**63),
            "--format",
            "npz",
            "--output",
            str(destination),
        ]
    )

    assert code == 1
    assert "min_flags must fit in a 64-bit signed integer" in capsys.readouterr().err
    assert not destination.exists()


def test_loaded_decisions_are_canonical_booleans(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    save_screen_archive(valid, _mixed_result())
    payload = _archive_payload(valid)
    # A Boolean byte of 2 compares as True, so it passes the decision checks.
    forged_members = {
        name: (payload[name].view(np.uint8) * 2).view(np.bool_)
        for name in ("flag__irv", "consensus_eligible", "consensus_flags")
    }
    forged = tmp_path / "forged.npz"
    _rewrite(valid, forged, **forged_members)

    restored = load_screen_archive(forged)["result"]

    for values in (
        restored["flags"]["irv"],
        restored["consensus_eligible"],
        restored["consensus_flags"],
    ):
        assert set(np.unique(values.view(np.uint8)).tolist()) <= {0, 1}
    _assert_same_result(restored, _mixed_result())
