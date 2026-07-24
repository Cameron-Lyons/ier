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

- **NumPy-first core.** Base installs depend only on NumPy. SciPy powers a
  small set of extras (`mahad` chi2/zscore flagging, `mahad_qqplot`,
  `response_time_mixture`, ML theta in `lz`). Matplotlib is only required for
  plotting helpers.
- **Registry.** `src/ier/_registry.py` maps string names to scorers, default
  screen/composite membership, flag direction, and required `IndexOptions`
  fields (e.g. `evenodd_factors`).
- **Soft per-index errors.** `score_registered_indices()` catches validation /
  runtime failures per index and returns them in an `errors` dict instead of
  aborting the whole screen. Callers still raise if *no* index succeeds where
  that is required (composite).

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
| Presence | `onset` | Any detected changepoint is flagged |

Threshold defaults are **sample-relative heuristics**, not calibrated
diagnostic cutoffs. See [Threshold Guidance](thresholds.md).

## Missing data

Most indices honor `IndexOptions.na_rm` (default `True`) and document
row-wise vs listwise behavior in their docstrings. Policies are intentionally
not identical across indices: IRV can use `nanstd`, Mahalanobis may drop
incomplete rows, Markov may require complete sequences depending on `na_rm`.
When comparing to R packages, align NA policy first.

## Composite scores

`composite()` z-combines selected indices with direction multipliers so that
higher composite values mean more evidence of careless responding.
`composite_probability()` applies a logistic transform for convenience — it is
**not** a calibrated probability of carelessness. Do not treat it as a
posterior or diagnostic probability without your own validation study.

## Response-time helpers

Timing matrices are a different data modality than item responses. Helpers in
`response_time.py` are public but **intentionally outside** the screen /
composite registry so they are not mixed into item-response pipelines by
accident.

## Optional dependencies

Install hints are centralized in `_optional_imports.py`:

- SciPy → `pip install 'insufficient-effort[full]'`
- matplotlib → `pip install 'insufficient-effort[plot]'`

## Parity and simulation

- Hand-locked regression fixtures live in `tests/test_golden_parity.py` and
  JSON under `tests/fixtures/parity/`.
- Detection-rate simulation: `benchmarks/bench_detection.py`.
- Throughput microbench: `benchmarks/bench_screen.py`.
