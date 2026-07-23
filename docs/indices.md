# Index Catalog

Registry-backed indices can be selected in `screen()` / `composite()`. Response-time
helpers use a different input domain and are listed separately.

## Matrix indices

| Name | Construct | Flag when | Screen default | Composite | Extra config |
|------|-----------|-----------|----------------|-----------|--------------|
| `irv` | Intra-individual response variability | low | yes | yes | — |
| `longstring` | Max consecutive identical responses | high | yes | yes | — |
| `longstring_pattern` | Repeating response patterns | high | yes | yes | `longstring_max_pattern_length` |
| `mahad` | Mahalanobis distance (multivariate outlier) | high | yes | yes | — |
| `psychsyn` | Psychometric synonym consistency | low | yes | yes | `psychsyn_critval` |
| `psychant` | Psychometric antonym consistency | low | no | yes | `psychant_critval` |
| `person_total` | Extreme total scores | low* | yes | yes | — |
| `markov` | Transition entropy | low | yes | yes | — |
| `u3_poly` | Polytomous person-fit / Guttman-like | high | yes | no | `scale_min` / `scale_max` |
| `midpoint` | Midpoint responding | high | yes | no | `scale_min` / `scale_max`, `midpoint_tolerance` |
| `acquiescence` | Agreeing / yea-saying | high | yes | no | scale bounds; optional item lists |
| `guttman` | Guttman errors | high | yes | yes | `guttman_normalize` |
| `individual_reliability` | Split-half individual reliability | low | no | yes | `reliability_n_splits`, seed |
| `onset` | Carelessness onset item index | present | no | no | `onset_window_size`, `onset_min_items` |
| `evenodd` | Even-odd consistency | low | no | yes | `evenodd_factors` |
| `mad` | Maximum absolute deviation (antonyms) | high | no | yes | MAD item lists / `mad_scale_max` |
| `lz` | lz person-fit | low | no | yes | optional IRT params via direct API |
| `semantic_syn` | Predefined synonym consistency | low | no | yes | `semantic_item_pairs` |
| `semantic_ant` | Predefined antonym consistency | low | no | yes | `semantic_item_pairs` |
| `infrequency` | Failed attention / bogus items | high | no | yes | item indices + expected responses |

\* `person_total` flags unusually low totals under the default low-direction
percentile rule; interpret in context of your scale coding.

## Response-time indices (standalone)

Call these with timing matrices (`respondents × items` or per-respondent times):

| Function | Signal | Typical flag |
|----------|--------|--------------|
| `response_time` | Central tendency of RT | low (too fast) |
| `response_time_consistency` | RT coefficient of variation | low (too uniform) |
| `response_time_flag` | Percentile / threshold flagging | low |
| `response_time_mixture` | Mixture P(fast component); needs SciPy | high |

## Plot helpers

Requires `insufficient-effort[plot]`:

- `plot_distributions(screen_result)`
- `plot_flag_counts(screen_result)`
- `plot_flagged_heatmap(screen_result)`
- `mahad_qqplot(...)` (also needs SciPy)
