# Contributing to IER

Thanks for contributing. This document covers local setup, quality checks, and release flow.

## Development Setup

### Recommended (`uv`)

```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
uv sync --all-groups
```

### Fallback (`pip`)

```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
python -m pip install -e . --group integration --group lint --group docs --group security
```

Development dependencies are split into `test`, `integration`, `lint`, `docs`,
`security`, and `artifact` groups. `integration` includes the test group plus pandas and
Polars compatibility checks; `uv sync --all-groups` installs the complete
contributor environment.

## Run Quality Checks

Run the unified suite before opening a PR:

```bash
./scripts/check.sh
```

The script first verifies that the editable project version in `uv.lock` matches
the Python distribution version of `pyproject.toml`. When `uv` is available,
it then synchronizes all locked dependency
groups and runs every command without further environment mutation. Without `uv`,
it uses tools from the active environment.

Skip the docs build with `SKIP_DOCS=1 ./scripts/check.sh`.

Or run checks individually:

```bash
uv run --no-sync pytest tests/ -v --cov=ier --cov-report=term-missing
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src/ier benchmarks scripts
uv run --no-sync python scripts/check_public_typing.py
uv run --no-sync mkdocs build --strict
```

CI selects and verifies each supported Python interpreter explicitly across
Linux, macOS, and Windows. Additional jobs run the complete test suite against
the declared minimum NumPy version, 1.26.0, on Python 3.11 and the newest release
within the supported range on Python 3.14, using locked integration tools.
Project/lock version consistency is checked even when the PR has the
`no-version-bump` label. Every test environment enforces warnings as errors and
the 95% coverage floor with branch measurement enabled. Superseded PR runs are
cancelled and jobs have time limits.
The aggregate Veto check evaluates every prerequisite in GitHub's `needs`
payload, including newly added jobs. Failures, cancellations, and skipped
required jobs block it; the version comparison retains its explicit skip rule.

### Lint roles

- **Ruff** is the linter, formatter, and static security scanner. The selected
  rules include pycodestyle, Pyflakes, isort, pyupgrade, Bugbear, simplify,
  type-checking, flake8-bandit security checks, and Pylint error/convention/warning rules.
- **mypy** performs strict static type checking for source, benchmarks, and
  release scripts. The public consumer gate checks valid calls in
  `tests/typing/composite_valid.py` and exact expected diagnostic codes and
  locations in `tests/typing/composite_invalid.py`; an unrelated failure cannot
  satisfy the negative checks.
- **actionlint** validates workflow definitions, expressions, matrix references,
  reusable workflow calls, and embedded scripts in CI. Its versioned container
  includes ShellCheck and pyflakes, and the reusable lint workflow makes this
  gate apply to pull requests and releases.

Pre-commit hooks use isolated tool environments and do not need a project
dependency. Install them with `uvx pre-commit install` when desired.

Optional benchmarks:

```bash
uv run python benchmarks/bench_screen.py
uv run python benchmarks/bench_detection.py
uv run python benchmarks/bench_flagging.py --structure opposite --scale 1.7976931348623157e308 --respondents 2 --missing-rate 0 --percentile 50
uv run python benchmarks/bench_orchestration.py --structure constant
uv run python benchmarks/bench_orchestration.py --structure near-constant --weighted
uv run python benchmarks/bench_orchestration.py --scale 1e300
uv run python benchmarks/bench_orchestration.py --weight-scale 1e308
uv run python benchmarks/bench_orchestration.py --no-standardize --scale 1e-300 --weight-scale 1e-300
uv run python benchmarks/bench_orchestration.py --method sum --no-standardize --scale 1e-161 --weight-scale 1e-161 --respondents 1000
uv run python benchmarks/bench_orchestration.py --method max --no-standardize --scale 1e300 --weight-scale 1e-300
uv run python benchmarks/bench_orchestration.py --method sum --missing-rate 0 --weighted
uv run python benchmarks/bench_orchestration.py --method max --missing-rate 0 --weighted --min-valid-indices 2
uv run python benchmarks/bench_cli_output.py --format text --respondents 1000000 --top 10
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_markov.py --missing-rate 0.1
uv run python benchmarks/bench_guttman.py --respondents 1000 --items 1000 --structure continuous
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_sequence_scoring.py --missing-rate 0.1
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_sequence_scoring.py --respondents 100 --items 1000 --operations longstring longstring_pattern
uv run python benchmarks/bench_response_checks.py --checks 40
uv run python benchmarks/bench_cli_input.py --masks
uv run python benchmarks/bench_archive.py --respondents 100000 --indices 10
uv run python benchmarks/bench_archive.py --respondents 100000 --indices 10 --compress
uv run python benchmarks/bench_person_total.py --structure near-constant --order F
uv run python benchmarks/bench_person_total.py --structure categorical --scale 5e-324
uv run python benchmarks/bench_response_checks.py --checks 40 --dtype int64 --integer-offset 1152921504606846976 --missing-rate 0
uv run python benchmarks/bench_response_time.py --operation median --order F
uv run python benchmarks/bench_row_reductions.py --order F --missing-rate 0.1
uv run python benchmarks/bench_pair_differences.py --scale 2.5e307 --offset 2.5e307
uv run python benchmarks/bench_pair_differences.py --structure mixed-scale --scale 1.7976931348623157e308 --respondents 1000
uv run python benchmarks/bench_pair_differences.py --structure reflected-residual --scale 1e16
uv run python benchmarks/bench_evenodd.py --factors 1 --factor-items 80 --missing-rate 0.1
uv run python benchmarks/bench_reliability.py --order F --missing-rate 0
uv run python benchmarks/bench_lz.py --missing-rate 0.1
uv run python benchmarks/bench_lz.py --operation discrimination --missing-rate 0.1
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_mahad.py --na-rm --missing-row-rate 0.1
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_psychsyn.py --structure independent --missing-rate 0
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_psychsyn.py --operation psychsyn_critval --missing-mode scattered
```

All performance benchmarks share `benchmarks/_measurement.py`. Timed repeats
run with allocation tracing disabled; one additional call per operation measures
peak traced allocation in MiB. The peak is allocation during that call, not total
process memory. Input generation, imports, requested warmups, garbage collection,
and disposal of intermediate results stay outside the timing window. Results
are released after each call until the final repetition, whose outputs are kept
for correctness checks. Paired measurements alternate operation order; use at
least five repeats so retaining the final outputs cannot dominate the median.
Allocation samples remain alive until their peak is recorded and tracing stops,
keeping result cleanup out of the allocation report. Tracing stops even when a
measured call fails; already-active tracing is rejected to avoid reporting
distorted timings.

Use `benchmarks/bench_score_reuse.py --workflow response-time` to compare fresh
Gaussian-mixture scoring with archived timing reflagging. Both paths include
loading and NPZ output; their complete output members and independently resolved
percentile decisions are checked after measurement.

The orchestration benchmark checks component calibration, screening summaries
and coverage, and sampled weighted reductions after measurement. It accepts
`--structure`, `--scale`, and `--offset` to exercise decimal constants, nearby
scores, and extreme finite units, as well as `--missing-rate 1` for unavailable
components. `--warmup` defaults to one untimed composite and screening call.
Use `--weight-scale` for a common positive weight multiplier; it enables
nonuniform weights even without `--weighted`. Mean runs cover very large and
tiny weight units, including sparse availability. Every resolved weight must
remain positive and finite.
Sum and maximum runs validate the weighted total and winning product against the
same reference. Use subnormal products for sum repair checks; choose scales whose
final sum or maximum fits the finite float range.
Compare `--missing-rate 0` with partial and entirely missing components to check
complete-data accumulation and reused availability masks. `--min-valid-indices`
also exercises coverage counts without changing the prepared input vectors.

The item-pair benchmark compares sampled MAD, semantic, and balanced acquiescence
scores with exact Fraction sums and Decimal deviations outside measurement.
`--structure mixed-scale` uses symmetric bounds and inserts extreme marker rows
among small signed responses. It requires float64 inputs and zero offset;
unrepresentable MAD totals retain infinity and are checked against the reference.
`--structure reflected-residual` pairs a large scale endpoint with small
responses, checking residuals that can disappear during reverse scoring. It
requires float64, zero offset, and scale at least `2**53`.

The flagging benchmark validates fixed and percentile flags against a sorted
Decimal reference outside measurement. It supports constant, nearly constant,
and opposite-sign profiles through `--structure`, finite transformations through
`--scale` and `--offset`, and entirely missing scores through `--missing-rate 1`.

The Guttman benchmark accepts `--structure continuous` to exercise scoring with
many distinct values, `--order F` for column-contiguous responses, and
`--missing-rate 1` for entirely missing inputs. Its correctness checks compare
sampled respondents with the direct pair-count definition under the full sample's
item-difficulty ordering. Wide high-cardinality inputs use bounded sorted runs;
smaller widths and small categorical scales retain their existing counters.

Use the same Python, NumPy, BLAS thread count, and benchmark arguments for
before/after comparisons. Older benchmark results that included tracing in the
timer are not directly comparable; rerun both revisions with the same measurement
method. Lower reported times after this tooling change do not indicate a change
in scoring performance. The detection benchmark is a synthetic accuracy study
and does not measure runtime or allocation.

The sequence benchmark measures longstring, repeating-pattern, and Markov
indices alongside default screening and composite workflows. Compare with
`--missing-rate 0` to check complete-data performance.
The Markov benchmark accepts `--states`
to compare small response scales with its sparse high-cardinality path.
The response-check benchmark measures missing-response and attention-check
scoring, including applicability masks and all attention-check missing policies.
Use `--checks` to vary the number of selected items and `--order F` to compare
column-contiguous inputs. Add `--dtype int64|uint64 --missing-rate 0` and
`--integer-offset` to exercise exact category comparisons near 64-bit limits.
Offsets that move the five response categories outside the selected dtype are
rejected. Every operation checks sampled results against a direct Python
reference outside measurement, including missing policies and unavailable
applicability denominators.
For the person-fit benchmark, use
`--missing-rate 0` for complete responses, `--order F` for column-contiguous
inputs, `--categories 5` for polytomous responses instead of binary data, and
`--model 1pl` for Rasch scoring instead of the default 2PL model.
Use `--operation discrimination` to isolate 2PL item-discrimination estimation;
that measurement excludes binary-response preparation. The default `lz`
operation includes preprocessing, calibration, and respondent scoring.
The person-total benchmark supports `--order F`, `--missing-rate 1`, and
`--strict` for missing propagation. Use `--structure` to select continuous,
categorical, constant, or nearby-baseline responses, and `--scale`/`--offset`
to exercise extreme response units. Noncontinuous structures and transformed
response units use exact rational sample item means and Decimal Pearson
correlations for correctness checks outside measurement.
For Mahalanobis scoring, use `--na-rm` to measure complete-case handling on
complete data, `--missing-row-rate`
to mark a fraction of rows as incomplete, and `--order F` to compare layouts.
Missing rows automatically enable complete-case handling.
Use `--operation qq --items 2` to include theoretical chi-square quantiles and
observed-distance sorting; `--items 1` exercises the normal-distribution special case.
The psychometric synonym benchmark measures scoring, item discovery with
`--operation psychsyn_critval`, or the shared item-correlation kernel with
`--operation correlations`. Use `--structure independent` for sparse pair
selection, `--missing-mode scattered` to spread omissions across items, and
`--order F` for column-contiguous responses.

The even–odd benchmark accepts `--factor-items` and `--factors` to compare narrow
and wide factor definitions, and `--order F` for column-contiguous inputs. Its
correctness check accounts for respondents without enough complete item pairs,
including fully missing data.
The reliability benchmark also supports `--order F`, reports the number of
available scores, and accepts legitimately undefined split-half corrections.
The predefined-pair benchmark accepts `--order F`, `--missing-rate 1`, and
`--scale` / `--offset` to transform its five response categories. It checks score
availability against complete item pairs. Use `--scale 2.5e307 --offset 2.5e307`
to exercise reverse-scoring bounds whose sum overflows, or
`--scale 1e-15 --offset 1.1` for nearly constant responses.
Use `bench_row_reductions.py --irv-splits 10` to include section-averaged IRV.
Add `--dtype float32` to exercise single-precision inputs and `--strict` to
propagate missing responses in IRV and acquiescence.
Its correctness checks preserve unavailable scores, including entirely missing
respondents and respondents with an entirely missing IRV section.
Both row-reduction and reliability benchmarks accept `--structure constant` or
`--structure near-constant` to exercise decimal-valued profiles and rounding
repairs. Constant-profile checks require exactly zero variability and unavailable
split-half reliability.
For integer row reductions, use `--dtype int64 --missing-rate 0`, with
`--integer-offset 1152921504606846976` to test adjacent large values. Unsigned
64-bit responses are supported as well. Integer runs use the categorical
structure, check exact midpoint and endpoint proportions, and reject response
bounds outside the chosen dtype.
The pair-difference benchmark accepts the same integer dtype, missing-rate, and
integer-offset options for MAD, semantic consistency, and balanced acquiescence.
The psychometric benchmark also accepts these options. It quantizes its generated
responses before applying the exact integer offset; combine them with
`--operation correlations` to isolate item discovery.
The onset benchmark also accepts `--order F` and `--missing-rate 1`. Compare
larger windows with `--items 200 --window-size 50 --min-items 50`, and use
`--structure continuous` to exercise noncategorical responses. Integer runs use
`--dtype int64` or `--dtype uint64`, optionally with `--integer-offset`, and
require complete categorical data. Sampled integer scores are checked against
the original categories before applying that offset. Its checks
verify that detected positions are integer offsets within sufficiently long
observed response sequences; absent detections remain valid benchmark outcomes.
The response-time benchmark supports `--operation median` and `--order F`.
It verifies score availability, accepts `--missing-rate 1` for median-only runs,
and requires enough usable medians when benchmarking the mixture. Single-item
timing runs omit the consistency metric, which requires at least two items.
Use `--no-log-transform --scale 1e300` to exercise extreme raw-time mixture fits,
and `--structure constant` or `--structure near-constant` for degenerate samples.

Verify release artifacts after packaging changes:

```bash
uv build
uv run --locked --only-group artifact --python 3.14 python scripts/check_dist.py dist/*
uv run --isolated --no-project --with dist/*.whl python scripts/smoke_test_install.py
uv run --isolated --no-project --with dist/*.tar.gz python scripts/smoke_test_install.py
```

The artifact verifier checks all Python package files, typing support, license,
CLI entry point, and release metadata in the wheel and source distribution.
It parses runtime and optional dependency metadata with the tooling-only
`packaging` dependency, including markers and direct references. The source
distribution must also retain its test modules, fixtures, imported scripts,
benchmark modules, and `uv.lock` so its bundled tests can run with reproducible
tools.
Package sources must match the checkout byte for byte, and the source
distribution's entire bundled project table must match the current project,
including runtime dependencies, optional dependencies, and entry points.
The source distribution also preserves the complete build configuration,
manifest, README, and license byte for byte. Artifacts with stale package
modules, duplicate members, ambiguous metadata headers, or unsafe archive paths
are rejected; source distributions cannot contain links. Wheel console scripts
must match the complete declared script table, including additional commands.
Wheel validation also requires `WHEEL` and `RECORD` and verifies the complete
integrity inventory, secure hashes, and any recorded sizes against the ZIP
payloads. Alterations with valid ZIP checksums still fail when their recorded
digests disagree; hashes are checked in bounded chunks.
`WHEEL` must declare supported wheel format 1.0 and pure-Python tags matching
the filename, with consistent build metadata.
Both artifacts are installed into isolated environments
with only runtime dependencies, then tested with known screening scores,
compressed CSV input, JSON and NPZ CLI output, respondent IDs, and archive
reload/reflagging. Timing smoke checks also remove the original input, re-export
saved medians, and reflag them in place, preserving IDs and missing scores while
checking fixed-cutoff equality and percentile ties independently.
CI also downloads the built artifacts into a job with no repository checkout,
extracts the bundled source-distribution tests and their support files, and
installs locked integration tools without installing the source project. It
installs the wheel, verifies the package imports from that isolated environment,
then runs the complete bundled suite against the installed wheel with the same
warnings and coverage gates. This checks packaging omissions that an editable
checkout can hide.
Release and publish workflows reuse these tested artifacts.
Both workflows also require lint, type, docs, and dependency-audit checks to pass
before creating a release or publishing a package.

## Architecture

See [docs/architecture.md](docs/architecture.md) for registry design, flagging
policy, NA handling, and composite score caveats.

## Pull Request Expectations

- Add or update tests for behavioral changes.
- Keep public docs/examples aligned with API changes.
- Keep CI green (tests, lint, security, docs workflows).
- Open an issue first for large API or behavior changes.
- Version bumps in `pyproject.toml` are required when `src/` changes
  (docs/CI-only PRs do not need a bump; use the `no-version-bump` label to skip).

## Release Process

The repository supports two release paths:

- Tag-based GitHub release workflow (`vX.Y.Z`) — runs the full CI suite, then
  produces artifacts and a GitHub Release. The tag must exactly match
  `v<project.version>` from `pyproject.toml`.
- Publish workflow (`Publish to PyPI`) — runs tests, validates artifacts, then uploads to
  TestPyPI/PyPI. Release-triggered publishes enforce the same tag/version match.

### Publish to TestPyPI

1. Open GitHub Actions.
2. Run `Publish to PyPI` manually.
3. Set `target=testpypi`.

### Publish to PyPI

1. Open GitHub Actions.
2. Run `Publish to PyPI` manually with `target=pypi`,
   or publish a GitHub Release manually.

A release created by the tag workflow uses GitHub's default token, which does
not trigger the separate release-event publish workflow. After tagging, dispatch
`Publish to PyPI` manually when the package should be published.

## Versioning Policy

- Use semantic versioning (`MAJOR.MINOR.PATCH`).
- Prerelease project versions must also represent Python prereleases. Use
  `X.Y.Z-alpha.N`, `X.Y.Z-beta.N`, `X.Y.Z-rc.N`, or `X.Y.Z-dev.N`.
  Python aliases (`a`, `b`, `c`, `pre`, and `preview`), implicit stage zero,
  and a following `dev.N` suffix are also supported. Arbitrary SemVer stages
  such as `canary` cannot be built by setuptools. Numeric and `post` suffixes
  are excluded because Python treats them as later releases while SemVer
  treats them as prereleases.
- Tags and bundled source metadata keep the exact `project.version` spelling.
  Generated artifact paths, core version metadata, and the lock entry use its
  normalized Python version: `1.9.0-rc.1` becomes `1.9.0rc1`.
  Local build metadata normalizes case, separators, and numeric segments;
  omit `+build` metadata for public PyPI releases.
- Source-changing PRs must set a valid semantic version strictly greater than
  the version on `main`; CI rejects unchanged versions and downgrades.
- Keep the editable project version in `uv.lock` equal to the normalized
  Python distribution version of `project.version`;
  local and CI checks reject drift before dependency synchronization.
- Bump `/pyproject.toml` when preparing a new public package release.
- PyPI does not allow uploading new files for a version that already exists.
  If you need to correct packaging for the same code, publish a new patch version.
