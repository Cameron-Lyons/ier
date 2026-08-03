# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.15.0] - 2026-08-02

### Added

- `ier composite --include-components` now exposes successful raw per-index
  scores and respondent-level valid-index counts in text, JSON, CSV, and NPZ.
  The option reuses the detailed summary workflow without rescoring, retains
  existing failure metadata, streams JSON arrays and CSV rows, and writes typed
  NPZ members without stacking a component matrix. On 100,000 respondents and
  five indices, detailed CSV used 0.274 MiB peak temporary allocation, JSON used
  0.525 MiB, and NPZ used 0.899 MiB, without adding a dependency.

### Fixed

- Composite text ranking now excludes `NaN` and infinite aggregate scores
  instead of allowing unavailable rows to appear above finite scores.

## [2.14.1] - 2026-08-02

### Fixed

- `ier composite` now preserves the soft per-index failures already collected
  during scoring instead of silently discarding them, and `ier screen` mirrors
  its existing failures to standard error as well. Every output format emits a
  concise warning, text output includes an `errors` section, JSON includes an
  `errors` object, and NPZ stores aligned `error_names` and `error_messages`
  vectors. CSV remains an unchanged respondent table, score computation still
  runs once, and no dependency was added.

## [2.14.0] - 2026-08-02

### Added

- All composite APIs now accept an optional `min_valid_indices` requirement.
  Respondents with fewer available component scores receive `NaN` before
  flagging or logistic transformation, while the default preserves established
  reduction behavior. Detailed summaries expose respondent-aligned
  `valid_index_counts`, and the CLI accepts `--min-valid-indices` and records the
  rule in text, JSON, and NPZ metadata. Equal-weight means reuse their existing
  count vector: on 500,000 respondents and 20 indices, enabling a 10-index
  minimum held peak temporary allocation at 17.2 MiB and changed median time
  from 48.3 to 48.4 ms. The weighted path adds about 4.2 ms and one 3.8 MiB
  integer count vector only when the rule is enabled, without adding a
  dependency.

## [2.13.0] - 2026-08-02

### Added

- All composite APIs now accept optional positive finite per-index `weights`,
  applied after direction correction and optional standardization. Weighted
  means renormalize over the indices available for each respondent, partial
  mappings leave unspecified selected indices at weight 1, and resolved
  weights are included in composite summaries. The CLI accepts repeatable
  `--weight INDEX=VALUE` options and records explicit overrides in text, JSON,
  and NPZ metadata without adding a dependency. On a 500,000-respondent,
  20-index standardized mean, weighting adds about 4.5 ms and 3.9 MiB of
  temporary allocation over the equal-weight reducer.

### Changed

- Standardizing a constant index now retains its missing values instead of
  converting them to zero contributions, so per-respondent composite means can
  correctly exclude unavailable evidence.

## [2.12.2] - 2026-08-02

### Changed

- Screening flag counts and composite mean, sum, and maximum reductions now
  accumulate one index at a time instead of constructing additional
  respondent-by-index matrices. On 500,000 respondents and 20 indices, a
  standardized composite mean reduces peak temporary allocation from 252.3 MiB
  to 16.8 MiB and median reduction time from 79.9 to 42.2 ms; screening flag
  counts drop from 13.4 MiB to 3.9 MiB and from 5.5 to 3.7 ms, without adding a
  dependency.

## [2.12.1] - 2026-08-02

### Changed

- CLI screening, composite, and response-time JSON output now writes respondent arrays
  in bounded chunks to plain files, gzip files, or standard output while
  preserving the existing schema and strict `null` handling. For 100,000
  respondents and five indices, screening JSON reduces peak output allocation
  from 44.5 MiB to 0.6 MiB, median serialization time from 0.83 to 0.57 seconds,
  and output size from 20.4 to 12.8 MiB without adding a dependency.

## [2.12.0] - 2026-08-02

### Added

- `screen()` and all composite APIs now accept an opt-in `workers` count, and
  the corresponding CLI commands accept `--workers`. Independent indices run
  concurrently through the Python standard library while results and failures
  retain selection order, and duplicate index selections are rejected rather
  than being scored ambiguously. Four workers reduce default screening time
  from 86.4 to 42.8 ms for 20,000 respondents × 80 items and from 531.0 to
  282.5 ms for 100,000 × 80, without changing the single-worker default or
  dependencies.

## [2.11.1] - 2026-08-02

### Changed

- CLI matrix loading, text/JSON/CSV serialization, and command coordination now
  live in focused internal modules. The coordinator is roughly half its prior
  size while input memory use, serialization throughput, output schemas, and the
  NumPy-only runtime dependency remain unchanged.

## [2.11.0] - 2026-08-02

### Added

- CLI scoring commands now write versioned, pickle-free `.npz` result archives
  with typed scores, flags, metadata, failures, and optional respondent IDs. On
  a 100,000-respondent, five-index result, NPZ output is about 650 times faster
  than CSV and produces a 5.2 MiB file instead of 11.2 MiB, without adding a
  dependency.

## [2.10.1] - 2026-08-02

### Changed

- CLI CSV output now writes rows directly to plain files, gzip files, or
  standard output instead of retaining all rows and the complete serialized
  document. A 100,000-respondent, five-index export reduces peak allocation
  from 136.6 MiB to 0.3 MiB and median runtime from 1.93 to 1.81 seconds.

## [2.10.0] - 2026-08-02

### Added

- CLI scoring commands now accept uncompressed `.npy` matrices and memory-map
  them read-only for fast, low-overhead loading without new dependencies. A
  25,000-respondent, 80-column benchmark initializes mapped input in about
  0.3 ms with 0.3 MiB peak allocation, versus 343 ms and 16.4 MiB for CSV.

## [2.9.2] - 2026-08-02

### Changed

- CLI matrix loading now validates and converts rows directly into a compact
  numeric buffer instead of retaining a full raw string matrix during
  conversion. A 25,000-respondent, 80-column benchmark reduces peak allocation
  from about 100 MiB to 16 MiB and runs roughly 2.4 times as fast.

## [2.9.1] - 2026-08-02

### Added

- CLI scoring commands now accept `-` for forward-only standard input and
  explicit standard output. All CLI commands transparently read or write `.gz`
  files where applicable, using only the Python standard library.

## [2.9.0] - 2026-08-02

### Added

- The new `ier response-time` command scores timing matrices with mean, median,
  standard-deviation, minimum, consistency, or Gaussian-mixture metrics. It
  supports fixed or percentile flagging, named respondent and timing columns,
  mixture configuration, and text, JSON, or CSV output without new dependencies.

## [2.8.2] - 2026-08-02

### Changed

- Complete-data psychometric synonym and antonym scoring now evaluates
  respondent-by-pair workspaces in bounded batches. A 5,000-respondent,
  80-item dense-pair benchmark reduces peak allocation from about 512 MiB to
  10 MiB and runs roughly 1.2 times as fast.

## [2.8.1] - 2026-08-02

### Changed

- Complete-matrix lz person-fit scoring now batches safeguarded theta estimation
  and likelihood calculations in cache-sized workspaces instead of looping over
  respondents. The 10,000-respondent, 80-item benchmark runs roughly six times
  as fast, while missing-data rows retain the scalar fallback.

## [2.8.0] - 2026-08-02

### Added

- `ier screen` and `ier composite` now accept repeatable, comma-separated
  `--item-columns` selections. Named items are resolved from the input header in
  requested order, can be combined with `--id-column`, and allow survey exports
  containing unselected nonnumeric metadata to be scored directly.

## [2.7.4] - 2026-08-02

### Changed

- Resampled individual reliability now streams per-respondent correlation sums
  and valid counts instead of retaining a respondent-by-split matrix, uses a
  faster missing-free correlation path, and isolates seeded randomness from
  NumPy's global state. The 10,000-respondent, 80-item, 100-split benchmark runs
  roughly 2.6 times as fast with lower peak workspace.

## [2.7.3] - 2026-08-02

### Changed

- Complete-matrix carelessness-onset detection now evaluates sliding-window
  variability and changepoint statistics in bounded vectorized batches instead
  of nested respondent/window loops. The 10,000-respondent, 80-item benchmark
  runs over 200 times as fast while missing-data rows retain the established
  fallback behavior.

## [2.7.2] - 2026-08-02

### Changed

- Public percentile-capable flag helpers now share one cutoff-boundary policy:
  explicit thresholds include scores exactly at the cutoff, while
  sample-percentile thresholds continue to flag only the strict tail. This
  brings direct helper behavior in line with fixed thresholds in `screen()`.

## [2.7.1] - 2026-08-02

### Changed

- Complete-matrix repeating-pattern scoring now streams match lengths in reverse
  using respondent-sized, smallest-safe integer vectors instead of several
  respondent-by-position integer matrices; the 10,000-respondent, 80-item
  benchmark runs roughly 1.6 times as fast with substantially less workspace,
  reducing the same default-screen benchmark by about 10%.

## [2.7.0] - 2026-08-02

### Added

- `screen()` and all composite orchestration APIs accept `strict=True` to raise
  a contextual error as soon as any selected index fails.
- The `screen` and `composite` CLI commands expose the same policy through
  `--strict`, while soft per-index errors remain the default.

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
