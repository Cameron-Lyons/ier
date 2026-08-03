# Index Catalog

Registry-backed indices can be selected in `screen()` / `composite()`. Response-time
helpers use a different input domain and are listed separately.

Inspect the same registry metadata programmatically or from the command line:

```python
from ier import index_catalog

catalog = index_catalog()
print(catalog["evenodd"]["required_options"])
```

```bash
ier indices
ier indices --format json --output indices.json
```

The catalog reports flag direction and mode, screen/composite availability and
defaults, and options that must be configured before an index can run.

## Matrix indices

| Name | Construct | Flag when | Screen default | Composite | Extra config |
|------|-----------|-----------|----------------|-----------|--------------|
| `irv` | Intra-individual response variability | low | yes | yes | — |
| `longstring` | Max consecutive identical responses | high | yes | yes | — |
| `longstring_pattern` | Repeating response patterns | high | yes | yes | `longstring_max_pattern_length` |
| `mahad` | Mahalanobis distance (multivariate outlier) | high | yes | yes | — |
| `psychsyn` | Psychometric synonym consistency | low | yes | yes | `psychsyn_critval` |
| `psychant` | Psychometric antonym consistency | low | no | yes | `psychant_critval` |
| `person_total` | Agreement with the sample item profile | low* | yes | yes | — |
| `markov` | Transition entropy | low | yes | yes | — |
| `missing_rate` | Missing-response proportion | high | no | yes | optional item subset via direct API |
| `u3_poly` | Polytomous person-fit / Guttman-like | high | yes | no | `scale_min` / `scale_max` |
| `midpoint` | Midpoint responding | high | yes | no | `scale_min` / `scale_max`, `midpoint_tolerance` |
| `acquiescence` | Agreeing / yea-saying | high | yes | no | scale bounds; optional item lists |
| `guttman` | Guttman errors | high | yes | yes | `guttman_normalize` |
| `individual_reliability` | Split-half individual reliability | low | no | yes | `reliability_n_splits`, seed |
| `onset` | Carelessness onset item index | present | no | no | `onset_window_size`, `onset_min_items` |
| `evenodd` | Even-odd consistency | low | no | yes | `evenodd_factors` |
| `mad` | Maximum absolute deviation (antonyms) | high | no | yes | MAD item lists / `mad_scale_max` |
| `lz` | lz person-fit | low | no | yes | optional IRT params via direct API; overflow-safe logistic kernel |
| `semantic_syn` | Predefined synonym consistency | low | no | yes | `semantic_item_pairs` |
| `semantic_ant` | Predefined antonym consistency | low | no | yes | `semantic_item_pairs`, optional scale bounds |
| `infrequency` | Failed attention / bogus items | high | no | yes | item indices + expected responses |

\* `person_total` flags unusually low correlations with the sample-wide item
profile under the default low-direction percentile rule.

`individual_reliability(..., random_seed=...)` uses an isolated reproducible
random stream. It does not reset or advance NumPy's process-wide random state.

The registry's `longstring` index uses `longstring_scores()` for numeric response
matrices. The standalone `longstring()` helper analyzes text strings only and
rejects numeric or multidimensional arrays.

`semantic_ant` reverse-scores the second item in each configured pair before
computing consistency. Pass `scale_min` and `scale_max` through `IndexOptions`
when the matrix does not contain both response-scale endpoints; otherwise the
bounds are inferred from the observed data.

`missing_rate` is opt-in because planned skip logic and matrix preprocessing can
create legitimate omissions. Use the standalone function's `item_indices` option
to restrict the calculation to required items, or select it explicitly in
`screen()` / `composite()` when all matrix columns share a missingness policy.

## Response-time indices (standalone — not in the registry)

These helpers take **timing matrices** (durations), not item-response matrices.
They are intentionally excluded from `screen()` / `composite()` so item scores
and timestamps are never mixed by accident. Compute them separately and merge
flags in your analysis code if needed.

| Function | Signal | Typical flag |
|----------|--------|--------------|
| `response_time` | Central tendency of RT | low (too fast) |
| `response_time_consistency` | RT coefficient of variation | low (too uniform) |
| `response_time_flag` | Percentile / threshold flagging | low |
| `response_time_mixture` | Stable mixture P(fast component) | high |

## Plot helpers

Requires `insufficient-effort[plot]`:

- `plot_distributions(screen_result)`
- `plot_flag_counts(screen_result)`
- `plot_flagged_heatmap(screen_result)`
- `mahad_qqplot(...)`
