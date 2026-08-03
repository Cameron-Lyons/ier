# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.6.1] - 2026-08-02

### Fixed

- Percentile-based flagging APIs now consistently reject booleans, non-finite
  thresholds, and percentiles outside 0–100 instead of silently returning
  misleading flags.
- Response-time flagging now shares the same validated threshold implementation
  as response-matrix indices.

## [2.6.0] - 2026-08-02

### Added

- Public `missing_rate()` and `missing_rate_flag()` helpers quantify response
  omissions across all items or a validated item subset.
- Opt-in `missing_rate` registry support for screening, composites, CLI output,
  fixed thresholds, and index discovery without changing default workflows.

## [2.5.0] - 2026-08-02

### Added

- CLI `--id-column NAME` support excludes a named identifier column from scoring
  and preserves unique, nonblank respondent IDs in text, JSON, and CSV outputs.

## [2.4.3] - 2026-08-02

### Changed

- Guttman scoring now directly encodes bounded integer response scales and counts
  grouped category positions in vectorized passes; the 10,000-respondent,
  80-item categorical benchmark runs roughly four times as fast.

## [2.4.2] - 2026-08-02

### Changed

- Markov transition scoring now directly encodes bounded integer response scales
  and compacts unused states without sorting the full matrix; the
  10,000-respondent, 80-item index benchmark runs roughly twice as fast.

## [2.4.1] - 2026-08-02

### Changed

- Mahalanobis distance evaluation now uses a BLAS-backed matrix product for the
  quadratic form; the 10,000-respondent, 80-item index benchmark runs roughly
  31 times faster and default screening runs roughly twice as fast.

## [2.4.0] - 2026-08-02

### Added

- Public `index_catalog()` metadata for registered indices, including flagging
  modes, orchestration defaults, composite availability, and required options.
- `ier indices` discovery output in text, JSON, or CSV format.

### Changed

- Local and CI version checks now verify that the editable project entry in
  `uv.lock` matches `project.version` before dependency synchronization.

## [2.3.0] - 2026-08-02

### Added

- `screen()` accepts fixed per-index `thresholds` alongside percentile defaults
  and returns the actual cutoff applied for each successful index.
- The CLI accepts repeatable `--threshold INDEX=VALUE` options and includes
  applied thresholds in JSON and text output.

## [2.2.6] - 2026-08-02

### Changed

- Longest-run scoring now processes complete matrices column-wise instead of
  looping over respondents; the 10,000-respondent, 80-item index benchmark runs
  roughly 47 times faster and the 1,000-respondent, 50-item default screening
  benchmark runs roughly 1.6 times faster.

## [2.2.5] - 2026-08-02

### Changed

- Repeating-pattern scoring now evaluates complete matrices in vectorized batches;
  the 1,000-respondent, 50-item index benchmark runs roughly 36 times faster and
  default screening runs roughly three times faster.

## [2.2.4] - 2026-08-02

### Added

- `screen()` now returns configurable respondent-level `consensus_flags` alongside
  per-index flags and counts, with a default `min_flags=2` agreement threshold.
- The CLI exposes `--min-flags` and includes consensus decisions in text, JSON,
  and CSV output.

## [2.2.3] - 2026-08-02

### Changed

- Guttman scoring now counts ordered response pairs without materializing every
  respondent-by-item-pair value matrix, using an adaptive categorical fast path
  and bounded-memory fallback for high-cardinality data; the 3,000-respondent,
  100-item categorical benchmark uses roughly one-sixth the peak process memory
  and runs about 1.7 times faster.

### Fixed

- All-missing Guttman inputs now return missing scores without emitting empty-mean
  runtime warnings.

## [2.2.2] - 2026-08-02

### Changed

- Transition-entropy scoring now uses its vectorized batch implementation for
  complete matrices under the default missing-value policy; the 10,000-respondent,
  80-item transition benchmark is roughly five times faster.

## [2.2.1] - 2026-08-02

### Changed

- Repeating-pattern screening now precomputes constant-pattern and periodic-match
  runs, avoiding repeated sorting and rescanning for every candidate position;
  the 1,000-respondent, 80-item screen benchmark is roughly nine times faster.

### Fixed

- Architecture and index documentation no longer describes dependency-free
  statistical routines as requiring SciPy.

## [2.2.0] - 2026-07-29

### Added

- Dependency-free normal density/quantile helpers, bounded Newton IRT theta
  estimation, and chi-square quantiles backed by regularized incomplete-gamma
  series and continued-fraction calculations.
- SciPy-reference regression coverage for chi-square probabilities from
  `1e-12` through `1 - 1e-12` and 1–1,000 degrees of freedom.
- Targeted `test`, `integration`, `lint`, `docs`, and `security` dependency groups.

### Changed

- Mahalanobis chi-square and z-score flagging, Mahalanobis Q-Q data, lz theta
  estimation, and response-time mixtures now run consistently in the base install.
- The lz theta solver now converges to a tighter score-equation tolerance; locked
  lz scores can shift by roughly `1e-6` from SciPy's default bounded tolerance.
- Ruff now owns lint and static-security checks, including selected Pylint and
  flake8-bandit rules; CI jobs install only their required dependency groups.
- The legacy `full` extra is an empty compatibility alias.

### Removed

- SciPy and scipy-stubs, the redundant wheel build requirement, direct transitive
  pins for Pillow/Pygments/Requests, and unused pre-commit project dependency.
- Redundant Bandit and non-gating Pylint jobs and dependencies.

## [2.1.7] - 2026-07-28

### Fixed

- Semantic-antonym scoring now reflects reverse-keyed responses around the
  configured or inferred response scale, so consistent antonym pairs receive
  high consistency scores instead of being clipped as maximally inconsistent.

## [2.1.6] - 2026-07-27

### Fixed

- `longstring()` now rejects numeric and multidimensional NumPy arrays instead of
  silently converting them into meaningless character-run results; numeric survey
  matrices remain supported through `longstring_scores()`.

## [2.1.5] - 2026-07-27

### Fixed

- CLI matrix loading now treats blank delimited fields as missing values instead
  of rejecting the file, and header detection no longer discards a first data row
  whose first value is blank.

## [2.1.4] - 2026-07-27

### Fixed

- CLI CSV export now represents NaN and infinite scores as empty cells instead of
  non-numeric `nan` and `inf` strings.

## [2.1.3] - 2026-07-27

### Fixed

- Invalid multi-character, empty, or newline CLI delimiters now return a concise
  error instead of leaking a Python traceback.

## [2.1.2] - 2026-07-27

### Fixed

- CLI matrix loading now accepts the documented whitespace-delimited format,
  including mixed runs of spaces and tabs.

## [2.1.1] - 2026-07-27

### Fixed

- CLI analysis and output failures now return concise `error:` messages instead
  of leaking Python tracebacks for invalid user input.
- CLI JSON export now emits standards-compliant `null` values for NaN and
  infinite scores or summary statistics.

## [2.1.0] - 2026-07-24

### Added

- Architecture note (`docs/architecture.md`) covering registry design, flagging,
  NA policy, and uncalibrated composite probabilities.
- Expanded golden / R-parity fixtures for `guttman`, `markov`, `person_total`,
  `midpoint`, `lz`, and `onset`, plus JSON harness under `tests/fixtures/parity/`.
- Synthetic detection-rate benchmark (`benchmarks/bench_detection.py`).
- GitHub issue templates (bug / feature / methods) and a PR template.
- Shared SciPy install hints via `_optional_imports.require_scipy`.

### Changed

- Docs homepage quick start uses `IndexOptions` (2.0-compatible).
- `dev` extra composes `full`, `plot`, and `docs` instead of duplicating pins.
- Lint and security workflows also run on pushes to `main`.
- Screen throughput benchmark updated for the IndexOptions-only API.
- LICENSE copyright years updated through 2026.
- Package version bumped to 2.1.0.

### Removed

- Unused `raise_missing_config` flag on `score_registered_indices()` (soft-fail
  is the only orchestration path).

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
