# Threshold Guidance

There is no universal cutoff that separates attentive from careless responders
across all surveys. Thresholds depend on item content, scale length, incentive
structure, and base rate of IER.

## Recommended practice

1. **Prefer multi-index agreement.** Flag respondents who are extreme on several
   independent families (consistency, pattern, outlier, attention check) rather
   than a single index. `screen()` exposes this decision as `consensus_flags` and
   defaults to requiring two per-index flags; tune it with `min_flags`.
2. **Use sample-relative percentiles carefully.** Default `screen(..., percentile=95)`
   and `composite_flag(..., percentile=95)` are convenient starting points, not
   gold standards. With small *N*, percentiles are unstable.
3. **Reuse validated cutoffs.** When prior validation supports fixed cutoffs, pass
   them by index (for example, `screen(data, thresholds={"irv": 0.25,
   "longstring": 8})`). The result reports the actual cutoff used for every index.
4. **Anchor with designed checks.** Infrequency / instructed-response items give
   confirmatory evidence when available (`infrequency`).
5. **Inspect before excluding.** Review open-ended responses, timestamps, and
   substantive patterns before listwise deletion.
6. **Report sensitivity.** Show how results change under alternate cutoffs
   (e.g., 90th vs 95th vs 99th percentile).

Percentile-based public flagging helpers reject non-finite thresholds and
percentiles outside `[0, 100]` instead of silently returning misleading flags.
Their boundary rule matches `screen()`: fixed cutoffs are inclusive (at or
beyond the cutoff), while sample-percentile cutoffs use strict tail comparisons
so ties at the estimated percentile are not flagged.

## Literature-informed starting points

These are illustrative defaults from common practice, not package guarantees:

| Signal | Common starting rule | Notes |
|--------|----------------------|-------|
| Longstring | Flag very long consecutive identical strings | Depends on scale length and response options |
| IRV | Flag unusually low variability | Straightlining / near-straightlining |
| Psychometric synonyms | Low within-person synonym correlations | Needs enough correlated pairs |
| Even-odd | Low even-odd consistency | Requires known factor lengths |
| Infrequency | ≥1 failed attention check | Threshold of 1 is common for short batteries |
| Response time | Very fast page/item times | Absolute cutoffs are survey-specific |
| Mahalanobis | High multivariate distance | Direct chi² flagging and sample-relative screening percentiles are available |

See Curran (2016) and Meade & Craig (2012) for broader methodological discussion.

## Composite scores

Treat `composite()` / `composite_probability()` as **ranking tools** within a
sample. Do not interpret logistic composite values as calibrated probabilities
unless you validate them on labeled data from a similar context.
