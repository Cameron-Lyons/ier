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
| `missing_rate` | Missing-response proportion | high | no | yes | optional item subset / applicability mask |
| `u3_poly` | Proportion of extreme responses | high | yes | no | `scale_min` / `scale_max` |
| `midpoint` | Midpoint responding | high | yes | no | `scale_min` / `scale_max`, `midpoint_tolerance` |
| `acquiescence` | Agreeing / yea-saying | high | yes | no | scale bounds; optional equal-length polarity lists |
| `guttman` | Guttman errors | high | yes | yes | `guttman_normalize` |
| `individual_reliability` | Split-half individual reliability | low | no | yes | `reliability_n_splits`, seed |
| `onset` | Carelessness onset item index | present | no | no | `onset_window_size`, `onset_min_items` |
| `evenodd` | Even-odd consistency | low | no | yes | `evenodd_factors` |
| `mad` | Mean absolute paired difference | high | no | yes | MAD item lists / optional scale bounds |
| `lz` | lz person-fit | low | no | yes | optional IRT params via direct API; overflow-safe logistic kernel |
| `semantic_syn` | Predefined synonym consistency | low | no | yes | `semantic_item_pairs` |
| `semantic_ant` | Predefined antonym consistency | low | no | yes | `semantic_item_pairs`, optional scale bounds |
| `infrequency` | Failed attention / bogus items | high | no | yes | item indices, expected responses, missing policy |

\* `person_total` flags unusually low correlations with the sample-wide item
profile under the default low-direction percentile rule.

`irv(..., split=True)` averages population standard deviations across consecutive
sections, giving each section equal weight. Automatic splits distribute extra
items to the earliest sections and ignore empty sections when `num_split` exceeds
the item count. `num_split` must be a positive integer, including NumPy integer
scalars. Custom `split_points` must be strictly increasing integer positions
starting at zero and ending at the number of columns. With `na_rm=True`, missing
responses are omitted within each section; an entirely missing section leaves the
respondent's split score `NaN`. With `na_rm=False`, any missing response propagates.
Both modes accumulate in double precision, including for `float32` input, and
rescale exceptional finite values to preserve very large or small deviations.
Constant finite responses, including decimal values, have exactly zero
variability; nearly constant responses retain their representable differences.
Infinite responses leave their section deviations unavailable.

`onset` requires integer `window_size` and `min_items` values, including NumPy
integer scalars, with `window_size >= 2` and `min_items >= window_size`. It returns
`NaN` when no changepoint is detected, too few responses or running windows are
available, or a row contains an infinite response. With `na_rm=True`, missing
responses are removed in sequence order and returned positions are zero-based
within that observed sequence. `na_rm=False` rejects missing responses, including
in rows too short for detection. Large signed and unsigned integer responses
retain their adjacent differences instead of rounding to identical values
before variability is measured.

`individual_reliability(..., random_seed=...)` uses an isolated reproducible
random stream. It does not reset or advance NumPy's process-wide random state.
It averages valid split correlations and applies the Spearman–Brown correction
`2r / (1 + r)`. Corrected values can be below -1 and are at most 1. Respondents
without any valid split, or with a mean split correlation of exactly -1, receive
`NaN`; screening and composites treat these scores as unavailable. The standalone
`individual_reliability_flag()` helper continues to flag unavailable scores.
`n_splits` must be a positive integer, including NumPy integer scalars.
Constant respondent profiles remain unavailable, including decimal-valued
profiles whose floating-point mean requires rounding.

`evenodd` pairs alternating items within each configured factor and averages
the available respondent-level factor correlations. Factors need at least four
items to contribute two paired observations; an odd final item is unpaired.
Respondents without any valid factor correlation receive `NaN`, report a
diagnostic count of zero, and remain unavailable rather than being flagged.
Factor sizes must be positive integers whose sum matches the response columns.

`psychsyn` and `psychant` discover pairs from finite item correlations. Constant
items, items containing missing responses, and samples with fewer than two
respondents have undefined item correlations and cannot supply pairs, even at
a zero cutoff. At least two selected pairs are needed for a respondent score;
otherwise the score is `NaN` and remains unavailable in screening and composites.
Diagnostics still count a single observed pair, and resampling cannot replace
an insufficient number of selected pairs. Cutoffs must be finite real numbers;
`psychsyn_critval()` additionally requires a nonnegative minimum magnitude.
Item discovery preserves correlations at very large or small finite scales and
for nearly constant items with representable variation. Constant decimal-valued
items remain unavailable, and an infinite response invalidates its item.
Summaries with no available scores return `NaN` statistics without warnings.
Integer item discovery preserves small differences at large baselines. Person-total
profiles retain variation between nearly identical integer item means, while
Guttman scoring preserves exact integer difficulty rankings and response categories.

`acquiescence` uses the normalized respondent mean by default. For a balanced
instrument, pair positively and negatively worded items in order with
`IndexOptions.acquiescence_positive_items` and
`IndexOptions.acquiescence_negative_items`, or the matching CLI options. Lists
must be nonempty and equal in length so every configured item participates;
indices are 0-based positions in the scored matrix. Both item polarities use raw
agreement responses, without reversing negative items. Pair means are normalized
using `scale_min` and `scale_max`, so agreement with every item scores 1 and
disagreement with every item scores 0.

Response-style indices infer omitted scale bounds from observed responses and
require `scale_min <= scale_max`. If the data cannot supply an omitted bound,
`u3_poly`, `midpoint_responding`, and `acquiescence` return unavailable (`NaN`)
scores. Missing respondents remain unavailable even on a constant scale, where
acquiescence assigns 0.5 to respondents with a usable mean. Balanced-pair mode
requires both responses in a pair; `na_rm=False` propagates missing responses.
Acquiescence preserves normalization for very wide, subnormal, and nearly
constant finite scales, including adjacent large integer responses.
Midpoint tolerance must be finite and nonnegative.
Integer midpoint intervals retain adjacent categories even above the exact
integer range of double precision. Explicit response-scale bounds retain their
precision during comparisons with integer and single-precision responses.

`response_pattern()` calculates extreme-response and midpoint proportions, the
raw respondent mean, and population standard deviation in bounded batches,
sharing observed-response counts across all four summaries.
Large integer row means retain exact totals before final rounding, and
variability preserves small differences between adjacent large observations.

The registry's `longstring` index uses `longstring_scores()` for numeric response
matrices. The standalone `longstring()` helper analyzes text strings only and
rejects numeric or multidimensional arrays.

`semantic_ant` reverse-scores the second item in each configured pair before
computing consistency. Pass `scale_min` and `scale_max` through `IndexOptions`
when the matrix does not contain both response-scale endpoints; otherwise the
bounds are inferred from the observed data.

Explicit `item_pairs` for MAD and semantic consistency must contain exactly two
integer column indices per entry. Fractional values, Boolean values, and
out-of-range positions are rejected. Semantic pairs must refer to distinct
items; repeated pairs and ordering are preserved.

`mad` also reverse-scores the second item in each pair. Provide
`mad_scale_min` and `mad_scale_max` when observed responses may omit a scale
endpoint or use fractional endpoints. Higher MAD values mean greater paired
inconsistency. Positive and negative item lists must contain the same number of
items; mismatched lists are rejected rather than truncated. The standalone
`semantic_syn_flag()` and `semantic_ant_flag()`
helpers flag unusually low consistency scores.
Pair scoring avoids overflowing reverse-score endpoint sums and intermediate
differences for large finite responses. Semantic consistency can retain a finite
normalized score even when the unnormalized mean difference exceeds float range.
Large integer pairs preserve differences before floating-point conversion.
Floating pairs preserve the rounding residual of reverse scoring, so swapping
pair endpoints does not erase a small difference. Semantic normalization happens
before exceptionally small means receive their final rounding.

`missing_rate` is opt-in because planned skip logic and matrix preprocessing can
create legitimate omissions. Use `IndexOptions.missing_item_indices` to restrict
registry scoring to a fixed required-item subset, or pass the same subset as
`item_indices` to the standalone helper. For respondent-specific skip logic,
provide a Boolean `missing_applicable_mask` through `IndexOptions` or
`applicable_mask` directly. False cells are excluded from both the numerator and
denominator; rows without applicable selected items return `NaN` and are not
flagged. The CLI exposes fixed subsets through `--missing-item-indices`.

`infrequency` preserves its historical missing-response behavior with
`missing="pass"`: unanswered checks do not count as failures and remain in a
proportion denominator. Choose `"fail"` for conservative scoring, `"omit"` for
available-case proportions, or `"propagate"` to require complete attention-check
data. Configure registry scoring with `IndexOptions.infrequency_missing` and the
CLI with `--infrequency-missing`. The standalone `infrequency_flag()` can flag
either counts or proportions.

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
| `response_time_score_flags` | Reflag retained direct or mixture scores | low or high |

Response-time medians omit missing items and return `NaN` for entirely missing
respondents. Middle pairs preserve finite medians at very large or small scales;
`float32` pairs are averaged in double precision and integer pairs are averaged
before rounding the result to a float.

`response_time_mixture()` uses positive, finite respondent medians and leaves
other respondents' probabilities `NaN`. Its `n_components` must be an integer
of at least two, including NumPy integer scalars, and cannot exceed the number
of usable medians.
Constant usable medians return `1 / n_components`, since the timings do not
distinguish a fast group. Fits preserve finite probabilities at very large timing
scales, including with `log_transform=False`. The variance floor remains `1e-10`
in the fitted units (log timing by default, raw timing otherwise), so changing raw
timing units can affect fits whose variation is near that floor. Components with
negligible fitted mass cannot be selected as the fast group.

## Plot helpers

Requires `insufficient-effort[plot]`:

- `plot_distributions(screen_result)`
- `plot_flag_counts(screen_result)`
- `plot_flagged_heatmap(screen_result)`
- `mahad_qqplot(...)`
