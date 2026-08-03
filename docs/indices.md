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
| `longstring_pattern` | Repeating response patterns | high | yes | yes | `longstring_max_pattern_length`; bounded complete/missing batches |
| `mahad` | Mahalanobis distance (multivariate outlier) | high | yes | yes | — |
| `psychsyn` | Psychometric synonym consistency | low | yes | yes | `psychsyn_critval`, retry seed; bounded complete and pairwise-complete correlations |
| `psychant` | Psychometric antonym consistency | low | no | yes | `psychant_critval`, retry seed; bounded complete and pairwise-complete correlations |
| `person_total` | Agreement with the sample item profile | low* | yes | yes | — |
| `markov` | Transition entropy | low | yes | yes | bounded dense/sparse and retained-length batches |
| `missing_rate` | Missing-response proportion | high | no | yes | optional item subset / applicability mask |
| `u3_poly` | Polytomous person-fit / Guttman-like | high | yes | no | `scale_min` / `scale_max` |
| `midpoint` | Midpoint responding | high | yes | no | `scale_min` / `scale_max`, `midpoint_tolerance` |
| `acquiescence` | Agreeing / yea-saying | high | yes | no | scale bounds; optional ordered polarity lists |
| `guttman` | Guttman errors | high | yes | yes | adaptive bounded counters; `guttman_normalize` |
| `individual_reliability` | Split-half individual reliability | low | no | yes | `reliability_n_splits`, seed; bounded stable raw moments |
| `onset` | Carelessness onset item index | present | no | no | `onset_window_size`, `onset_min_items` |
| `evenodd` | Even-odd consistency | low | no | yes | `evenodd_factors`; bounded row correlations and exact two-pair shortcut |
| `mad` | Mean absolute paired difference | high | no | yes | MAD item lists / optional scale bounds |
| `lz` | lz person-fit | low | no | yes | optional IRT params via direct API or `IndexOptions`; bounded, active-row-compacted complete/missing calibration; overflow-safe logistic kernel |
| `semantic_syn` | Predefined synonym consistency | low | no | yes | `semantic_item_pairs` |
| `semantic_ant` | Predefined antonym consistency | low | no | yes | `semantic_item_pairs`, optional scale bounds |
| `infrequency` | Failed attention / bogus items | high | no | yes | item indices, expected responses, missing policy |

\* `person_total` flags unusually low correlations with the sample-wide item
profile under the default low-direction percentile rule.

`individual_reliability(..., random_seed=...)` uses an isolated reproducible
random stream. It does not reset or advance NumPy's process-wide random state.

`IndexOptions` exposes calibrated LZ parameters as `lz_difficulty`,
`lz_discrimination`, and `lz_theta`, with `lz_model` selecting `"1pl"` or
`"2pl"`. The CLI equivalents accept parameter vector files. Any omitted array
or file retains the direct function's fallback estimate.

`psychsyn(..., diag=True)` and `psychant(..., diag=True)` return each score with
the number of selected item pairs actually available for that respondent.
Item-pair discovery uses all pairwise-complete observations, so isolated
omissions no longer remove an entire item from the analysis. A respondent needs
at least three usable pairs for a score; seeded retries keep the observed pair
count unchanged. Shared workflows accept independent `psychsyn_random_seed` and
`psychant_random_seed` values, with matching command-line options.

Fit psychometric pairs once on a reference cohort and reuse the immutable
calibration on later matrices with the same item count and column order:

```python
from ier import (
    IndexOptions,
    fit_psychsyn_model,
    load_psychsyn_model,
    psychsyn_model_scores,
    save_psychsyn_model,
    screen,
)

model = fit_psychsyn_model(reference_responses, critval=0.6)
save_psychsyn_model("psychsyn-model.npz", model)
model = load_psychsyn_model("psychsyn-model.npz")
later_scores, usable_pairs = psychsyn_model_scores(
    later_responses,
    model,
    diag=True,
)
calibrated_screen = screen(
    later_responses,
    indices=["psychsyn", "irv"],
    options=IndexOptions(psychsyn_model=model),
)
```

Fixed-model scoring never rediscovers pairs. The model retains the fitted
threshold, synonym/antonym mode, item count, and a read-only owned pair array.
Its versioned NPZ archive is pickle-free, strictly validated, and atomically
replaced. The same fit-and-score boundary is available from the command line:

```bash
ier psychsyn-fit reference.csv psychsyn-model.npz --critval 0.6
ier psychsyn-score later.csv psychsyn-model.npz --format json --output scores.json
ier screen later.csv --indices psychsyn irv --psychsyn-model psychsyn-model.npz
ier composite later.csv --indices psychsyn irv --psychsyn-model psychsyn-model.npz
```

Add `--antonym` when fitting an antonym model; its negative default threshold
is `-0.6`. Scoring infers the correct registered index and low-tail direction
from the saved mode. Named item selection must use the same order during both
commands because the model deliberately stores positions, not source headers.
Shared workflows use `psychsyn_model` or `psychant_model` on `IndexOptions` and
the corresponding CLI flags. A fixed model takes precedence over the matching
programmatic discovery threshold. On the CLI, supplying both is rejected as a
configuration conflict. Model type, synonym/antonym mode, and item count follow
the same per-index soft-failure or `strict=True` policy as other scorer errors;
immutable models can be shared safely by parallel workers.

`mahad_qqplot()` generates dependency-free theoretical chi-square coordinates
in bounded vector batches. Plotting remains optional; with `plot=False`, large
Q-Q datasets require only the NumPy base installation.

The registry's `longstring` index uses `longstring_scores()` for numeric response
matrices. The standalone `longstring()` helper analyzes text strings only and
rejects numeric or multidimensional arrays. Numeric longest-run and repeating-
pattern scoring remove missing responses in order, group rows by retained length,
and reuse bounded complete-response kernels.

`semantic_ant` reverse-scores the second item in each configured pair before
computing consistency. Pass `scale_min` and `scale_max` through `IndexOptions`
when the matrix does not contain both response-scale endpoints; otherwise the
bounds are inferred from the observed data.

`mad` also reverse-scores the second item in each pair. Provide
`mad_scale_min` and `mad_scale_max` when observed responses may omit a scale
endpoint or use fractional endpoints. Higher MAD values mean greater paired
inconsistency. The standalone `semantic_syn_flag()` and `semantic_ant_flag()`
helpers flag unusually low consistency scores.

`acquiescence` supports balanced polarity pairs through
`IndexOptions.acquiescence_positive_items` and
`IndexOptions.acquiescence_negative_items`. CLI screening accepts the same
zero-based ordered lists through `--acquiescence-positive-items` and
`--acquiescence-negative-items`; both must be supplied together.

`missing_rate` is opt-in because planned skip logic and matrix preprocessing can
create legitimate omissions. Use `IndexOptions.missing_item_indices` to restrict
registry scoring to a fixed required-item subset, or pass the same subset as
`item_indices` to the standalone helper. For respondent-specific skip logic,
provide a Boolean `missing_applicable_mask` through `IndexOptions` or
`applicable_mask` directly. False cells are excluded from both the numerator and
denominator; rows without applicable selected items return `NaN` and are not
flagged. The CLI exposes fixed subsets through `--missing-item-indices` and
respondent-specific 0/1 or Boolean mask files through
`--missing-applicable-mask`.

`infrequency` preserves its historical missing-response behavior with
`missing="pass"`: unanswered checks do not count as failures and remain in a
proportion denominator. Choose `"fail"` for conservative scoring, `"omit"` for
available-case proportions, or `"propagate"` to require complete attention-check
data. Configure registry scoring with `IndexOptions.infrequency_missing` and the
CLI with `--infrequency-missing`. The standalone `infrequency_flag()` can flag
either counts or proportions.

## Summary helpers

Import `mahad_summary()`, `markov_summary()`, and `psychsyn_summary()` directly
from `ier` for compact distribution, coverage, and method-specific diagnostics.
Their return contracts are exposed as `MahadSummary`, `MarkovSummary`, and
`PsychsynSummary`, so editors and static type checkers can discover every field
without falling back to an unstructured dictionary. Calculations and missing-data
semantics match the corresponding scoring functions. If every score is unavailable,
the distribution fields are `NaN` while the coverage fields report zero valid scores;
this is a normal result and does not emit a runtime warning.

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
| `fit_response_time_mixture` | Reusable reference-cohort mixture calibration | — |
| `response_time_mixture_scores` | Apply fixed mixture calibration | high |
| `response_time_score_flags` | Reflag retained direct or mixture scores | low or high |
| `flag_consensus` | Combine aligned flags after separate scoring | configurable count |

Median summaries and mixture preprocessing remove missing observations within
bounded equal-length row groups before median selection. This keeps large timing
matrices allocation-bounded while preserving each respondent's observation order.

`flag_consensus()` can then combine those timing flags with registered screen
flags while using optional score vectors to count per-respondent availability.
It does not accept or score either source matrix.

Fitted mixture models copy their small parameter vectors into read-only storage;
later cohorts can reuse the same calibration without repeating EM.

## Plot helpers

Requires `insufficient-effort[plot]`:

- `plot_distributions(screen_result)`
- `plot_flag_counts(screen_result)`
- `plot_flagged_heatmap(screen_result)`
- `mahad_qqplot(...)`
