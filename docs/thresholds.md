# Threshold Guidance

There is no universal cutoff that separates attentive from careless responders
across all surveys. Thresholds depend on item content, scale length, incentive
structure, and base rate of IER.

## Recommended practice

1. **Prefer multi-index agreement.** Flag respondents who are extreme on several
   independent families (consistency, pattern, outlier, attention check) rather
   than a single index. `screen()` exposes this decision as `consensus_flags` and
   defaults to requiring two per-index flags; tune it with `min_flags`.
2. **Require enough evaluable signals.** When missing inputs can make component
   scores unavailable, set `min_valid_indices` so consensus is withheld rather
   than inferred from too little evidence. Inspect `valid_index_counts` and
   `consensus_eligible` in the result. For separately scored domains, pass aligned
   flags and optional score vectors to `flag_consensus()` and use
   `min_valid_signals` for the equivalent rule.
3. **Use sample-relative percentiles carefully.** Default `screen(..., percentile=95)`
   and `composite_flag(..., percentile=95)` are convenient starting points, not
   gold standards. With small *N*, percentiles are unstable. When signal families
   need different sensitivities, pass per-index `percentiles` rather than forcing
   one tail setting across the entire screen.
4. **Reuse validated cutoffs.** When prior validation supports fixed cutoffs, pass
   them by index (for example, `screen(data, thresholds={"irv": 0.25,
   "longstring": 8})`). The result reports the actual cutoff used for every index.
5. **Anchor with designed checks.** Infrequency / instructed-response items give
   confirmatory evidence when available (`infrequency`).
6. **Inspect before excluding.** Review open-ended responses, timestamps, and
   substantive patterns before listwise deletion.
7. **Report sensitivity.** Show how results change under alternate cutoffs
   (e.g., 90th vs 95th vs 99th percentile).

Percentile-based public flagging helpers reject non-finite thresholds and
percentiles outside `[0, 100]` instead of silently returning misleading flags.
Their boundary rule matches `screen()`: fixed cutoffs are inclusive (at or
beyond the cutoff), while sample-percentile cutoffs use strict tail comparisons
so ties at the estimated percentile are not flagged.

Screen percentile overrides use the same directional convention as the global
setting: a value `p` resolves high-direction indices at `p` and low-direction
indices at `100-p`. Fixed and percentile overrides are mutually exclusive for an
index. The returned `threshold_sources` and `percentiles` mappings make the
resolved rule auditable alongside the numeric `thresholds`.

For designed attention checks, `infrequency_flag()` accepts either count cutoffs
or proportional cutoffs with `proportion=True`. State the chosen missing-response
policy alongside the cutoff because `pass`, `fail`, `omit`, and `propagate` encode
different assumptions about unanswered checks.

The composite command exposes the same rule through mutually exclusive
`--threshold` and `--percentile` options. Flagging is opt-in: without either
option, composite output contains scores only. The resolved cutoff and its
fixed or percentile source are retained in self-describing output formats.

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
unless you validate them on labeled data from a similar context. The CLI's
`--include-probability` option adds this logistic value for export only;
`--threshold` and `--percentile` continue to make decisions from the original
composite scores.
