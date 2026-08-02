# Screening Workflow

`screen()` runs multiple IER indices, applies flagging rules, and returns a
structured result.

Configure with a single `IndexOptions` object:

```python
from ier import IndexOptions, screen

result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
print(result["indices_used"])
print(result["flag_counts"])
print(result["consensus_flags"])
print(result["errors"])
```

## Result keys

| Key | Meaning |
|-----|---------|
| `scores` | Per-index score arrays |
| `flags` | Per-index boolean flags |
| `thresholds` | Actual per-index cutoffs (`None` for presence flagging) |
| `flag_counts` | Total flags per respondent |
| `consensus_flags` | Respondents meeting the configured multi-index agreement threshold |
| `min_flags` | Number of per-index flags required for consensus (default: 2) |
| `indices_used` | Successfully computed indices |
| `errors` | Soft failures (missing config, invalid data for an index) |
| `summary` | Mean/std/min/max/`n_flagged` per index |
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
        semantic_item_pairs=[(0, 1), (2, 3)],
        infrequency_item_indices=[9],
        infrequency_expected_responses=[5],
    ),
)
```

Missing required config is recorded in `result["errors"]` instead of aborting
the whole screening run. `composite()` uses the same soft-fail policy.

For production batches that require every requested index to succeed, enable
strict mode. The first failed index raises a contextual `ValueError`:

```python
result = screen(data, indices=["irv", "mad"], options=options, strict=True)
```

## Missing responses

Missing-response rate is available as an opt-in registry index:

```python
result = screen(data, indices=["missing_rate"], thresholds={"missing_rate": 0.2})
```

It is not a default because planned skip logic can create legitimate omissions.
Call `missing_rate(data, item_indices=[...])` directly when only required items
should contribute to the rate.

## Flagging

- Most indices use percentile thresholds (`percentile=95` by default).
- High-direction indices flag above the percentile; low-direction indices flag
  below `100 - percentile`.
- Pass fixed cutoffs with `thresholds={"irv": 0.25, "longstring": 8}`. Fixed
  thresholds are inclusive: high-direction scores at or above the cutoff and
  low-direction scores at or below the cutoff are flagged. Other indices keep
  using the configured percentile.
- `onset` uses presence flagging: any detected changepoint is flagged.
- `consensus_flags` marks respondents flagged by at least `min_flags` indices.
  Use `screen(..., min_flags=1)` for single-index workflows.

## Response times (out of band)

`response_time*` helpers take **timing matrices** (seconds or other duration
units), not Likert item responses. They are intentionally **not** registered in
`screen()` / `composite()` because mixing domains would silently mis-score
respondents. Call them directly and combine flags yourself if needed:

```python
from ier import response_time, response_time_flag

median_rt = response_time(times, metric="median")
flags = response_time_flag(times, cutoff_percentile=5)
```

## CLI

Blank fields in comma-, tab-, or semicolon-delimited input are loaded as missing
values (`NaN`) and follow each index's documented missing-data behavior.

```bash
ier screen data.csv --scale-min 1 --scale-max 5 --indices irv longstring
ier screen data.csv --min-flags 3
ier screen data.csv --threshold irv=0.25 --threshold longstring=8
ier screen data.csv --indices irv mad --strict
ier screen data.csv --format json --output screen.json
ier screen data.csv --format csv --evenodd-factors 5,5 --indices evenodd irv
ier --version
```

JSON output is standards-compliant: unavailable or non-finite scores and summary
statistics are encoded as `null`. CSV output represents non-finite scores as empty
cells so numeric columns remain compatible with spreadsheet and statistics tools.
