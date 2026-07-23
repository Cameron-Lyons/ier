# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- CI coverage gate aligned to 90%; Codecov upload fails the job on error;
  Bandit no longer falls back to a looser severity filter.
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
