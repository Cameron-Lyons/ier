# Screening Workflow

`screen()` runs multiple IER indices, applies flagging rules, and returns a
structured result.

Configure with a single `IndexOptions` object:

```python
from ier import IndexOptions, screen

result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
print(result["indices_used"])
print(result["flag_counts"])
print(result["valid_index_counts"])
print(result["consensus_flags"])
print(result["errors"])
```

## Result keys

| Key | Meaning |
|-----|---------|
| `scores` | Per-index score arrays |
| `flags` | Per-index boolean flags |
| `thresholds` | Actual per-index cutoffs (`None` for presence flagging) |
| `threshold_sources` | `fixed`, `percentile`, or `presence` provenance per cutoff |
| `percentiles` | Requested tail percentile per sample-relative cutoff; otherwise `None` |
| `flag_counts` | Total flags per respondent |
| `valid_index_counts` | Available index scores per respondent |
| `consensus_eligible` | Whether each respondent meets the optional completeness rule |
| `consensus_flags` | Respondents meeting the configured multi-index agreement threshold |
| `min_flags` | Number of per-index flags required for consensus (default: 2) |
| `min_valid_indices` | Required available scores for consensus, or `None` |
| `indices_used` | Successfully computed indices |
| `errors` | Soft failures (missing config, invalid data for an index) |
| `summary` | Mean/std/min/max, valid/unavailable counts, flagged count, and valid-score flag rate per index |
| `n_respondents` / `n_indices` | Size metadata |

## Defaults

Default indices are NumPy-only and require no extra item metadata:

`irv`, `longstring`, `longstring_pattern`, `mahad`, `psychsyn`, `person_total`,
`markov`, `u3_poly`, `midpoint`, `acquiescence`, `guttman`

Mahalanobis distances and all direct flagging methods (`"chi2"`, `"iqr"`, and
`"zscore"`) are available in the NumPy-only base install.

## Config-gated indices

Pass configuration to include indices that need survey metadata:

```python
from ier import IndexOptions, screen

result = screen(
    data,
    indices=["evenodd", "mad", "semantic_syn", "infrequency"],
    options=IndexOptions(
        evenodd_factors=[5, 5],
        mad_positive_items=[0, 1, 2],
        mad_negative_items=[3, 4, 5],
        mad_scale_min=1,
        mad_scale_max=5,
        semantic_item_pairs=[(0, 1), (2, 3)],
        infrequency_item_indices=[9],
        infrequency_expected_responses=[5],
        infrequency_missing="fail",
    ),
)
```

Missing required config is recorded in `result["errors"]` instead of aborting
the whole screening run. `composite()` uses the same soft-fail policy.

## Consensus completeness

An unavailable component score is not a flag. When incomplete item data or a
soft-failed index could leave a consensus decision supported by too few signals,
set a minimum availability requirement:

```python
result = screen(
    data,
    indices=["irv", "longstring", "infrequency"],
    options=options,
    min_flags=2,
    min_valid_indices=3,
)

reviewable = result["consensus_eligible"]
```

Rows below the minimum remain unflagged even if their available signals meet
`min_flags`. The result always reports `valid_index_counts` and
`consensus_eligible`; with the default `min_valid_indices=None`, every row is
eligible and prior consensus behavior is preserved. Both counts accumulate one
index vector at a time rather than constructing a respondent-by-index matrix.

```bash
ier screen data.csv --indices irv longstring infrequency \
  --infrequency-item-indices 3 --infrequency-expected-responses 5 \
  --min-flags 2 --min-valid-indices 3 --format json
```

### Attention-check missing responses

`infrequency` exposes four explicit missing-response policies through
`IndexOptions.infrequency_missing`, the standalone `missing=` argument, and CLI
`--infrequency-missing`:

| Policy | Behavior |
|--------|----------|
| `pass` | Treat missing checks as correct; legacy default |
| `fail` | Treat missing checks as failures |
| `omit` | Exclude missing checks from proportions; no observed checks yields `NaN` |
| `propagate` | Return `NaN` when any configured check is missing |

Set `infrequency_proportion=True` (or `--infrequency-proportion`) to score the
failure share instead of the count. `infrequency_flag(..., proportion=True)`
supports the same policy with an inclusive cutoff between zero and one.

For production batches that require every requested index to succeed, enable
strict mode. The first failed index raises a contextual `ValueError`:

```python
result = screen(data, indices=["irv", "mad"], options=options, strict=True)
```

## Parallel index scoring

Each selected index reads the same validated matrix independently. Larger
multi-index runs can opt into standard-library worker threads:

```python
result = screen(data, workers=4)
```

```bash
ier screen responses.npy --workers 4 --format npz --output screening.npz
```

The default is `workers=1`. Parallel results, soft failures, and strict failures
retain selection order. Additional workers can improve throughput when NumPy
kernels release the interpreter lock, but their temporary workspaces may overlap
and raise peak memory. Benchmark representative matrix sizes and worker counts.

## Reusing computed scores

Threshold and consensus sensitivity checks do not need to recalculate expensive
indices. Pass the score mapping from one screening run to `screen_scores()`:

```python
from ier import screen, screen_scores

initial = screen(data, indices=["irv", "longstring", "mahad"])

lenient = screen_scores(initial["scores"], percentile=90, min_flags=1)
strict = screen_scores(
    initial["scores"],
    percentiles={"irv": 99, "longstring": 99, "mahad": 99},
    min_flags=2,
)
```

The reusable path accepts registered index names, preserves their mapping order,
and returns the same structured result contract as `screen()`. All vectors must
be one-dimensional, non-empty, equally sized, and contain only finite values or
`NaN`. Compatible `float64` arrays are retained by reference and never mutated.
Because no scorer runs, the returned `errors` mapping is empty; retain the first
run separately if its soft failures are part of the audit record.

On the bundled 10,000-respondent, 80-item benchmark, evaluating five tail
percentiles from retained scores takes 5.1 ms and 1.6 MiB peak temporary
allocation instead of 175.7 ms and 21.5 MiB for five full runs, a 34.2x speedup
without another dependency.

Persist the same raw score mapping directly or through CLI NPZ output and reload
it later:

```python
from ier import load_score_archive, save_score_archive, screen_scores

save_score_archive("screening-scores.npz", result["scores"], errors=result["errors"])
saved = load_score_archive("screening.npz")
revised = screen_scores(saved["scores"], percentile=99, min_flags=3)
```

The public writer can also store aligned respondent IDs. The loader returns
those identifiers and any recorded soft failures.

## Missing responses

Missing-response rate is available as an opt-in registry index:

```python
options = IndexOptions(missing_item_indices=[0, 1, 4, 5])
result = screen(
    data,
    indices=["missing_rate"],
    options=options,
    thresholds={"missing_rate": 0.2},
)
```

It is not a default because planned skip logic can create legitimate omissions.
Use `missing_item_indices` when the same item subset is required for every
respondent. The option applies through `screen()` and all composite helpers;
the direct equivalent is `missing_rate(data, item_indices=[...])`.

For respondent-specific branching, supply a Boolean matrix matching the response
matrix. True cells identify expected responses and false cells are excluded from
both the missing count and denominator:

```python
import numpy as np

applicable = np.array(
    [
        [True, True, False, False],
        [True, True, True, True],
    ]
)
options = IndexOptions(missing_applicable_mask=applicable)
result = screen(data, indices=["missing_rate"], options=options, min_flags=1)
```

Rows without any applicable selected items receive `NaN` and are not flagged.
`missing_rate_flag()` accepts the same `applicable_mask` argument for direct
flagging.

## Flagging

- Most indices use percentile thresholds (`percentile=95` by default).
- High-direction indices flag above the percentile; low-direction indices flag
  below `100 - percentile`.
- Override individual tail settings with
  `percentiles={"irv": 90, "longstring": 99}`. Values use the same directional
  convention as the global setting, and unspecified indices retain the global
  percentile.
- Pass fixed cutoffs with `thresholds={"irv": 0.25, "longstring": 8}`. Fixed
  thresholds are inclusive: high-direction scores at or above the cutoff and
  low-direction scores at or below the cutoff are flagged. Other indices keep
  using the configured percentile.
- An index cannot have both fixed and percentile overrides. Presence-mode indices
  accept neither. Results retain every cutoff's source and requested percentile.
- `onset` uses presence flagging: any detected changepoint is flagged.
- `consensus_flags` marks respondents flagged by at least `min_flags` indices.
  Use `screen(..., min_flags=1)` for single-index workflows.

## Response times (out of band)

`response_time*` helpers take **timing matrices** (seconds or other duration
units), not Likert item responses. They are intentionally **not** registered in
`screen()` / `composite()` because mixing domains would silently mis-score
respondents. Call them directly or use the dedicated CLI command:

```python
from ier import (
    load_response_time_archive,
    response_time,
    response_time_flag,
    response_time_score_flags,
    save_response_time_archive,
)

median_rt = response_time(times, metric="median")
flags = response_time_flag(times, cutoff_percentile=5)
stricter_flags = response_time_score_flags(median_rt, cutoff_percentile=1)

saved = load_response_time_archive("timing.npz")
revised_flags = response_time_score_flags(
    saved["scores"],
    threshold=1.0,
    direction=saved["flag_direction"],
)
save_response_time_archive(
    "revised-timing.npz",
    saved["scores"],
    revised_flags,
    threshold=1.0,
    metric=saved["metric"],
    flag_direction=saved["flag_direction"],
    respondent_ids=saved["respondent_ids"],
    threshold_source="fixed",
)
```

```bash
ier response-time timings.csv --metric median --percentile 5
ier response-time timings.csv --metric consistency --threshold 0.05 --format csv
ier response-time timings.csv --metric mixture --components 2 --random-seed 42
ier response-time timings.csv --metric median --format npz --output timing.npz
```

Direct timing metrics and consistency scores use low-tail flagging. Mixture
probabilities use high-tail flagging. Fixed thresholds include equality; derived
percentile cutoffs exclude ties, matching the other public flagging workflows.
Retained direct scores use `direction="low"` by default; pass `direction="high"`
for mixture probabilities. This sensitivity path never recomputes row summaries
or refits the mixture. The NPZ loader also validates that archived flags agree
with their stored threshold, suspicious-tail direction, and—when present—fixed
or percentile provenance before reuse. Provenance-aware archives retain the
requested percentile and recompute the derived cutoff during validation. The
matching writer performs the same checks before creating a CLI-compatible
archive; calls without provenance remain compatible with legacy v1 output.
Mixture fitting excludes respondents whose median time is missing, infinite, or
non-positive. Its posterior normalization remains stable when ordinary Gaussian
density calculations underflow for an extreme valid observation.

## CLI

Blank fields in comma-, tab-, or semicolon-delimited input are loaded as missing
values (`NaN`) and follow each index's documented missing-data behavior.

```bash
ier screen data.csv --scale-min 1 --scale-max 5 --indices irv longstring
ier screen data.csv --min-flags 3
ier screen data.csv --min-flags 2 --min-valid-indices 3
ier screen data.csv --threshold irv=0.25 --threshold longstring=8
ier screen data.csv --index-percentile irv=90 --index-percentile longstring=99
ier screen data.csv --indices irv mad --strict
ier screen data.csv --format json --output screen.json
ier screen data.csv --format csv --evenodd-factors 5,5 --indices evenodd irv
ier screen data.csv --indices missing_rate --missing-item-indices 0,1,4
ier screen data.csv --indices infrequency \
  --infrequency-item-indices 3,7 \
  --infrequency-expected-responses 5,1 --infrequency-missing fail
ier screen data.csv --id-column participant_id --item-columns q1,q2,q3,q4
ier response-time timings.csv --metric median --threshold 1.0
ier screen data.csv.gz --format json --output screening.json.gz
ier screen data.npy --indices irv longstring --format json
ier screen data.npy --indices irv longstring --format npz --output screening.npz
cat data.csv | ier screen - --indices irv longstring --format json
ier --version
```

`--item-columns` accepts comma-separated header names and may be repeated. It
lets screen and composite commands ignore unselected metadata columns while
preserving the requested item order. Any item-index options refer to that
selected order.

Uncompressed `.npy` input is memory-mapped read-only and must contain one
non-empty, two-dimensional, real numeric array. It has no headers, so
`--id-column`, `--item-columns`, and `--delimiter` do not apply. Compressed
`.npy.gz` input is not supported because it cannot be memory-mapped.

Use `-` as the data path to read a forward-only standard-input stream, and use
`--output -` to select standard output explicitly. Input and output paths ending
in `.gz` are compressed or decompressed transparently with no optional package.
CSV rows and JSON respondent arrays are emitted in bounded chunks across all
three destinations.

JSON output is standards-compliant: unavailable or non-finite scores and summary
statistics are encoded as `null`. CSV output represents non-finite scores as empty
cells so numeric columns remain compatible with spreadsheet and statistics tools.
NPZ output preserves those values and typed flags exactly; its versioned layout is
documented in [CLI output formats](../cli-output.md).
