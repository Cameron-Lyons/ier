# Architecture

This note explains how IER orchestrates indices, handles missing data, and
flags respondents. It is aimed at contributors and methods-curious users.

## Layers

```text
CLI / examples
      │
screen() / composite()     ← public orchestration
      │
IndexOptions + registry    ← shared config + index catalog
      │
per-index modules          ← irv, longstring, mahad, …
      │
_validation / _flagging    ← shared input checks and threshold helpers
```

- **NumPy-first core.** Base installs depend only on NumPy. Statistical
  routines, including chi-square flagging, response-time mixtures, and IRT
  theta estimation, are implemented locally. Matplotlib is only required for
  plotting helpers.
- **Registry.** `src/ier/_registry.py` maps string names to scorers, default
  screen/composite membership, flag direction, and required `IndexOptions`
  fields (e.g. `evenodd_factors`).
- **Soft per-index errors.** `score_registered_indices()` catches validation /
  runtime failures per index and returns them in an `errors` dict instead of
  aborting the whole screen. Public orchestration APIs also accept `strict=True`
  to raise a contextual error as soon as a selected index fails. Composite
  callers still raise if *no* index succeeds under the default soft policy.
- **Opt-in parallelism.** Registry orchestration evaluates indices sequentially
  by default. `workers>1` uses a lazily imported standard-library thread pool,
  then records scores and failures in selection order. This keeps default import
  cost and resource use stable while allowing NumPy-heavy scorers to overlap.
- **Bounded reductions.** Screening flag counts and composite mean, sum, and
  maximum values accumulate one index at a time. Orchestration retains its
  documented per-index result vectors but does not construct another complete
  respondent-by-index matrix for final reductions.

## Command-line boundaries

The command-line path is split by responsibility:

- `cli.py` defines arguments, converts index options, and coordinates commands.
- `_cli_input.py` owns forward-only delimited input, gzip handling, named-column
  selection, and memory-mapped NumPy input.
- `_cli_composite.py` validates shared respondent alignment and flag metadata
  contracts for every composite serializer.
- `_cli_output.py` renders text plus bounded strict JSON and CSV results.
- `_cli_npz.py` writes versioned, typed, pickle-free NumPy result archives.

Screen and composite commands carry the registry's ordered soft-failure map
through text, JSON, and NPZ serializers and mirror failures to standard error
for every format. Score computation still runs once, and CSV remains a compact
respondent table.

Detailed composite output is explicit through `--include-components`. The CLI
reuses `composite_summary()` so scoring still runs once, JSON wraps each
component in the bounded array writer, CSV emits one row at a time, and NPZ
writes separate typed members without stacking another respondent-by-index
matrix. The default aggregate-only path does not allocate or serialize these
details.

Composite standardization is resolved before either aggregate-only or detailed
scoring begins. The same boolean is passed to the public scoring API and written
to text, JSON, and NPZ metadata, avoiding a second calculation or inference from
the resulting values. CSV intentionally remains a respondent-only table.

Optional composite flagging runs after aggregate scoring and reuses that score
vector. Cutoff resolution and boolean comparison use the same shared helpers as
the public flagging APIs. JSON writes the flag vector in bounded chunks, CSV
streams it row by row, and NPZ stores it as a boolean member; no component is
rescored and the score-only path allocates no flag vector.

Keeping parsing, matrix construction, serialization, and orchestration separate
makes format-specific changes independently testable without adding runtime
packages or coupling them to statistical index implementations.

## IndexOptions

Orchestration APIs accept configuration **only** via
`options=IndexOptions(...)`. That keeps `screen()` / `composite()` signatures
stable as indices grow. Per-function kwargs remain available when calling an
index module directly (e.g. `mahad(x, method="iqr")`).

## Flagging policy

`screen()` turns scores into boolean flags using registry metadata:

| Mode | Typical indices | Rule |
|------|-----------------|------|
| Low / high percentile | IRV, longstring, mahad, … | Extreme tail relative to the sample |
| Fixed low / high threshold | Any percentile-mode index | At or beyond a validated cutoff |
| Presence | `onset` | Any detected changepoint is flagged |

Percentile defaults are **sample-relative heuristics**, not calibrated
diagnostic cutoffs. `screen(thresholds=...)` accepts fixed cutoffs when a survey
or validation study provides them. See [Threshold Guidance](thresholds.md).

## Missing data

Most indices honor `IndexOptions.na_rm` (default `True`) and document
row-wise vs listwise behavior in their docstrings. Policies are intentionally
not identical across indices: IRV can use `nanstd`, Mahalanobis may drop
incomplete rows, Markov may require complete sequences depending on `na_rm`.
When comparing to R packages, align NA policy first.

## Composite scores

`composite()` z-combines selected indices with direction multipliers so that
higher composite values mean more evidence of careless responding.
Optional positive weights are applied after direction correction and
standardization. Weighted means renormalize over available scores per
respondent, so an unavailable index does not silently dilute the remaining
evidence.
An optional minimum valid-index rule masks under-supported respondent scores
after reduction. Equal-weight means reuse their existing denominator counts, so
the rule needs no additional respondent-sized workspace on that path; other
methods allocate one integer count vector only when the rule is enabled.
`composite_probability()` applies a logistic transform for convenience — it is
**not** a calibrated probability of carelessness. Do not treat it as a
posterior or diagnostic probability without your own validation study.

## Response-time helpers

Timing matrices are a different data modality than item responses. Helpers in
`response_time.py` are public but **intentionally outside** the screen /
composite registry so they are not mixed into item-response pipelines by
accident.

## Optional dependencies

All statistical functionality is available in the NumPy-only base install.
Plotting remains optional and reports a centralized install hint from
`_optional_imports.py`: `pip install 'insufficient-effort[plot]'`.

## Parity and simulation

- Hand-locked regression fixtures live in `tests/test_golden_parity.py` and
  JSON under `tests/fixtures/parity/`.
- Detection-rate simulation: `benchmarks/bench_detection.py`.
- Throughput microbench: `benchmarks/bench_screen.py`.
- Screen/composite reduction memory: `benchmarks/bench_orchestration.py`.
- CLI JSON, CSV, and NPZ serialization: `benchmarks/bench_cli_output.py`.
