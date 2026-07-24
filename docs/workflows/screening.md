# Screening Workflow

`screen()` runs multiple IER indices, applies flagging rules, and returns a
structured result.

Configure with a single `IndexOptions` object:

```python
from ier import IndexOptions, screen

result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
print(result["indices_used"])
print(result["flag_counts"])
print(result["errors"])
```

## Result keys

| Key | Meaning |
|-----|---------|
| `scores` | Per-index score arrays |
| `flags` | Per-index boolean flags |
| `flag_counts` | Total flags per respondent |
| `indices_used` | Successfully computed indices |
| `errors` | Soft failures (missing config, invalid data for an index) |
| `summary` | Mean/std/min/max/`n_flagged` per index |
| `n_respondents` / `n_indices` | Size metadata |

## Defaults

Default indices are NumPy-only and require no extra item metadata:

`irv`, `longstring`, `longstring_pattern`, `mahad`, `psychsyn`, `person_total`,
`markov`, `u3_poly`, `midpoint`, `acquiescence`, `guttman`

Mahalanobis distances in screening use a NumPy-safe path; SciPy is only needed
when you call `mahad(..., flag=True, method="chi2"|"zscore")` directly.

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

## Flagging

- Most indices use percentile thresholds (`percentile=95` by default).
- High-direction indices flag above the percentile; low-direction indices flag
  below `100 - percentile`.
- `onset` uses presence flagging: any detected changepoint is flagged.

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

```bash
ier screen data.csv --scale-min 1 --scale-max 5 --indices irv longstring
ier screen data.csv --format json --output screen.json
ier screen data.csv --format csv --evenodd-factors 5,5 --indices evenodd irv
ier --version
```
