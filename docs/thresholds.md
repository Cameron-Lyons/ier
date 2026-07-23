# Threshold Guidance

There is no universal cutoff that separates attentive from careless responders
across all surveys. Thresholds depend on item content, scale length, incentive
structure, and base rate of IER.

## Recommended practice

1. **Prefer multi-index agreement.** Flag respondents who are extreme on several
   independent families (consistency, pattern, outlier, attention check) rather
   than a single index.
2. **Use sample-relative percentiles carefully.** Default `screen(..., percentile=95)`
   and `composite_flag(..., percentile=95)` are convenient starting points, not
   gold standards. With small *N*, percentiles are unstable.
3. **Anchor with designed checks.** Infrequency / instructed-response items give
   confirmatory evidence when available (`infrequency`).
4. **Inspect before excluding.** Review open-ended responses, timestamps, and
   substantive patterns before listwise deletion.
5. **Report sensitivity.** Show how results change under alternate cutoffs
   (e.g., 90th vs 95th vs 99th percentile).

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
| Mahalanobis | High multivariate distance | chi² flagging needs SciPy; screening uses percentiles |

See Curran (2016) and Meade & Craig (2012) for broader methodological discussion.

## Composite scores

Treat `composite()` / `composite_probability()` as **ranking tools** within a
sample. Do not interpret logistic composite values as calibrated probabilities
unless you validate them on labeled data from a similar context.
