# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-24

### Breaking

- `screen()` / `composite()` / `composite_flag()` / `composite_summary()` /
  `composite_probability()` accept configuration only via `options=IndexOptions(...)`.
  Legacy per-index keyword arguments were removed.
- Removed deprecated `build_index_options()`.
- String run-length helpers in `ier.longstring` are private
  (`_run_length_encode`, `_run_length_decode`, `_longstr_message`,
  `_avgstr_message`). Prefer `longstring()` / `longstring_scores()`.
- `mahad(..., flag=True, method="zscore")` requires SciPy (same as `chi2`);
  it no longer silently falls back to a hardcoded threshold of `2.0`.

### Added

- CLI JSON/CSV export (`--format json|csv`, optional `--output`), plus
  IndexOptions knobs (`--evenodd-factors`, MAD/semantic/infrequency lists, etc.).
- Clearer jagged-CSV error when loading CLI matrices.
- Golden parity fixtures for `longstring_pattern`, `mahad` (iqr), `psychsyn`,
  and `evenodd`.
- Public export of `longstring_scores`.
- `SECURITY.md` and `.github/CODEOWNERS`.
- Release and PyPI publish workflows run the full CI test suite before shipping.
- CI also runs on pushes to `main`.

### Changed

- `composite()` soft-fails missing index config (e.g. evenodd without factors),
  matching `screen()`; if no index succeeds, it still raises.
- Local `scripts/check.sh` pylint uses `--fail-under=9.0` to match CI.
- Version-check workflow is callable-only (no duplicate PR trigger).
- Response-time helpers documented as intentionally out of the registry.
- Specialized tests split into `test_composite.py`, `test_response_time.py`,
  and a smaller `test_specialized_indices.py`.
- Package version bumped to 2.0.0.

### Fixed

- Changelog no longer claims Codecov fails the job on upload error; uploads stay
  non-blocking when `CODECOV_TOKEN` is unset.

## [1.8.0] - 2026-07-23

### Added

- Public `IndexOptions` config object for `screen()` / `composite()` (preferred
  over long keyword lists); legacy kwargs remain supported when `options` is omitted.
- Package `__version__` and `ier` CLI (`ier screen`, `ier composite`).
- Unified contributor check script (`scripts/check.sh`).
- Golden IRV / longstring parity fixtures (`tests/test_golden_parity.py`) and a
  concrete R-package comparison table in the docs.
- Docs build gated on pull requests via the veto workflow.

### Changed

- `screen()` / `composite()` share the full `IndexOptions` surface (including
  scale bounds, longstring pattern length, onset, and acquiescence settings).
- `screen()` reuses shared `threshold_flags` for percentile flagging.
- `visualize` helpers are typed against `ScreenResult`.
- Version-check CI requires a bump only when `src/` changes.
- Pre-commit Ruff pin aligned to 0.15.x; docs workflow uses `actions/checkout@v7`.
- Package version bumped to 1.8.0.

### Fixed

- Release workflow no longer overwrites curated `CHANGELOG.md` when generating
  GitHub release notes.

## [1.7.0] - 2026-07-23

### Added

- Registry coverage for `guttman`, `psychant`, `individual_reliability`, `onset`,
  `semantic_syn`, `semantic_ant`, and `infrequency` in `screen()` / `composite()`.
- `onset` presence-based flagging mode in `screen()`.
- Expanded `screen()` / `composite()` configuration knobs for newly registered indices.
- MkDocs documentation site (getting started, workflows, index catalog, thresholds,
  R notes, API reference).
- Benchmark script for the `screen()` hot path (`benchmarks/bench_screen.py`).
- Example scripts under `examples/`.
- pandas / polars smoke tests; pandas and polars added to the `dev` extra.
- Curated changelog and expanded PyPI classifiers (Python 3.11–3.14).

### Changed

- Default `screen()` / `composite()` Mahalanobis scoring uses a NumPy-safe path
  (`method="iqr"` distances). SciPy is only required for direct
  `mahad(..., flag=True, method="chi2")` (and related SciPy-only helpers).
- `score_registered_indices` soft-catches `ValueError`, `RuntimeError`, and
  `TypeError` into per-index `errors`.
- CI coverage gate aligned to 90%; Codecov upload remains non-blocking when a
  token is unset; Bandit no longer falls back to a looser severity filter.
- Package version bumped to 1.7.0.

### Fixed

- Base-install footgun where default screening could abort on missing SciPy when
  computing Mahalanobis distances with the previous chi-squared default path.

## [1.6.3] - 2026-05

### Changed

- Dependency and CI maintenance releases (Dependabot, lockfile, Actions bumps).

### Added

- Index orchestration registry hardening and related workflow improvements.

## Earlier

Feature work through early 2026 introduced screening workflows, composite
scoring, additional indices (including MAD, lz, acquiescence), visualizations,
and multi-OS / multi-Python CI.
