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
defaults, options that must be configured before an index can run, and whether
the index reads reverse-scored responses (`uses_keyed_responses`; see
[Reverse-keyed items](#reverse-keyed-items)). Every
`required_options` entry must be set, and at least one option in each
`alternative_options` group: `infrequency` requires `infrequency_item_indices`
and either `infrequency_expected_responses` or `infrequency_acceptable_ranges`.
[CLI output formats](cli-output.md#index-catalog) describes the `ier indices`
text, JSON, and CSV layouts.

## Matrix indices

| Name | Construct | Flag when | Screen default | Composite | Extra config |
|------|-----------|-----------|----------------|-----------|--------------|
| `irv` | Intra-individual response variability | low | yes | yes | optional `irv_num_split` / `irv_split_points` |
| `longstring` | Max consecutive identical responses | high | yes | yes | — |
| `longstring_pattern` | Repeating response patterns | high | yes | yes | `longstring_max_pattern_length` |
| `mahad` | Mahalanobis distance (multivariate outlier) | high | yes | yes | — |
| `psychsyn` | Psychometric synonym consistency | low | yes | yes | `psychsyn_critval`, `psychsyn_item_correlations` |
| `psychant` | Psychometric antonym consistency | high | no | yes | `psychant_critval`, `psychsyn_item_correlations` |
| `person_total` | Agreement with the sample item profile | low* | yes | yes | — |
| `markov` | Transition entropy | low | yes | yes | — |
| `missing_rate` | Missing-response proportion | high | no | yes | optional item subset / applicability mask |
| `u3_poly` | Proportion of extreme responses (not PerFit's U3; see `u3poly_fit`) | high | yes | no | `scale_min` / `scale_max` |
| `midpoint` | Midpoint responding | high | yes | no | `scale_min` / `scale_max`, `midpoint_tolerance` |
| `acquiescence` | Agreeing / yea-saying | high | yes | no | scale bounds; optional equal-length polarity lists |
| `guttman` | Guttman errors | high | yes | yes | `guttman_normalize` |
| `individual_reliability` | Split-half individual reliability | low | no | yes | `reliability_n_splits`, seed, optional `reliability_factors` |
| `onset` | Carelessness onset item index | present | no | no | `onset_window_size`, `onset_min_items` |
| `evenodd` | Even-odd consistency | low | no | yes | `evenodd_factors`, `evenodd_method` |
| `mad` | Mean absolute paired difference | high | no | yes | MAD item lists / optional scale bounds |
| `lz` | lz person-fit | low | no | yes | optional IRT params via direct API; overflow-safe logistic kernel |
| `semantic_syn` | Predefined synonym consistency | low | no | yes | `semantic_item_pairs` |
| `semantic_ant` | Predefined antonym consistency | low | no | yes | `semantic_item_pairs`, optional scale bounds |
| `infrequency` | Failed attention / bogus items | high | no | yes | item indices, expected responses or acceptable ranges, missing policy |
| `avgstr` | Mean length of identical-response runs | high | no | yes | — |
| `autocorrelation` | Strongest lagged autocorrelation (cyclic patterns) | high | no | yes | `autocorrelation_max_lag`, `autocorrelation_statistic` |
| `gpoly` | Normalized polytomous Guttman errors over item steps (PerFit `Gnormed.poly`) | high | no | yes | optional `person_fit_ncat`, `scale_min` / `scale_max` |
| `u3poly_fit` | Polytomous U3 person fit over item steps (PerFit `U3poly`) | high | no | yes | optional `person_fit_ncat`, `scale_min` / `scale_max` |
| `ht` | Transposed scalability Ht for 0/1 items (PerFit `Ht`) | low | no | yes | dichotomous responses |

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
Split IRV (Dunn et al., 2018; `careless::irv(split = TRUE)`) is also available to
`screen()` and composites through `IndexOptions(irv_num_split=N)` or
`IndexOptions(irv_split_points=[...])`, and on the CLI through `--irv-num-split N`
or `--irv-split-points 0,10,20`. Split points override the section count. The
default single section keeps the unsplit computation, and invalid values are
reported as index failures rather than silently ignored.

`onset` requires integer `window_size` and `min_items` values, including NumPy
integer scalars, with `window_size >= 2` and `min_items >= window_size`. It returns
`NaN` when no changepoint is detected, too few responses or running windows are
available, or a row contains an infinite response. With `na_rm=True`, missing
responses are removed in sequence order and returned positions are zero-based
within that observed sequence. `na_rm=False` rejects missing responses, including
in rows too short for detection. Large signed and unsigned integer responses
retain their adjacent differences instead of rounding to identical values
before variability is measured.

`individual_reliability` never reads or advances NumPy's process-wide random
state. `random_seed` accepts an integer, which reproduces the established split
sequence, or a `np.random.Generator`, which the call advances; without a seed,
fresh entropy is used. `IndexOptions.reliability_random_seed` takes integers.

With `factors=` (`IndexOptions.reliability_factors`, CLI
`--reliability-factors 8,8,8`), each resample randomly splits every scale into
two halves of `size // 2` items, as in resampled individual reliability (Curran,
2016; Huang et al., 2012). A respondent's two vectors of scale half means are
correlated across scales, valid correlations are averaged over resamples, and
the Spearman–Brown correction `2r / (1 + r)` is applied and clamped below at -1.
Each half mean is its exact mean rounded once, as in `evenodd`, so a split whose
half means do not vary is unusable rather than scored from rounding noise.
At least two scales are required, single-item scales contribute no halves, and
respondents with fewer than two complete half-mean pairs in every resample
receive `NaN`. This scale-aware form is recommended whenever the scale structure
is known: in a simulated six-scale survey it separates uniform random responders
from attentive respondents with an AUC near 0.99, where the legacy form is near
chance.

Without `factors`, the legacy form correlates randomly paired items across the
whole questionnaire. Items of one scale share a trait level, so these
correlations carry little signal for attentive respondents. It averages valid
split correlations and applies the Spearman–Brown correction. Corrected values
can be below -1 and are at most 1. Respondents without any valid split, or with
a mean split correlation of exactly -1, receive `NaN`. In both forms, screening
and composites treat unavailable scores as unavailable, and the standalone
`individual_reliability_flag()` helper continues to flag them. `n_splits` must be
a positive integer, including NumPy integer scalars. Constant respondent profiles
remain unavailable, including decimal-valued profiles whose floating-point mean
requires rounding.

`evenodd` offers two algorithms through `method=` (`IndexOptions.evenodd_method`,
CLI `--evenodd-method`). `method="halves"` is the classic even-odd index
(Johnson, 2005; Meade & Craig, 2012; Curran, 2016) and is recommended. Each factor
is split into the mean of its odd-position items and the mean of its
even-position items, ignoring missing responses. The two vectors of factor half
means are correlated within each respondent across the factors where both halves
are available, and the correlation is Spearman–Brown corrected and clamped below
at -1. Higher scores mean more consistent responding, so the score equals
`-careless::evenodd`. Each half mean is its exact mean rounded once, as R's
extended-precision `mean()` computes it, so half means with equal exact values are
identical: a respondent whose decimal-valued half means do not vary is unavailable,
as in careless, rather than scored from rounding noise. In simulated comparisons the
only remaining differences from careless, below 1e-7, are respondents whose half
means vary only in their last binary digit; there R's long-double `cor()` keeps
rounding error that the correlation here avoids. At least two factors are
required. Respondents with fewer than two available factors, or whose half means
do not vary across factors, receive `NaN`; straight-lined profiles are therefore
unavailable here and are covered by `longstring`. The `diag=True` count reports
the factors whose two half means are both finite.

`method="item_pairs"` remains the default for compatibility. It pairs
alternating items within each configured factor and averages the available
respondent-level factor correlations. Factors need at least four items to
contribute two paired observations; an odd final item is unpaired. Respondents
without any valid factor correlation receive `NaN`, report a diagnostic count of
zero, and remain unavailable rather than being flagged. Because the items of a
factor share a trait level, these within-factor correlations are close to
uninformative for attentive respondents. In either mode, factor sizes must be
positive integers whose sum matches the response columns.

Recode reverse-worded items with `reverse_score(x, items, scale_min, scale_max)`
before `evenodd`, `individual_reliability`, `guttman`, `lz`, `gpoly`, `u3poly_fit`,
or `ht`, which assume all items are keyed in the same direction (the indices
whose `uses_keyed_responses` catalog field is true). It returns a copy in which
each selected response `v` becomes `scale_min + scale_max - v`, inferring missing
bounds from the observed minimum and maximum of the whole matrix. Integer and Boolean responses
with integral bounds are recoded exactly in an integer dtype: the input dtype when
it holds both bounds, otherwise the smallest integer dtype that holds the input
dtype's range and both bounds. Unsigned 64-bit responses with a negative bound have
no such dtype; they, integer responses with a non-integral bound, and other inputs
return float64 with each exact reflection rounded once, including integer responses
beyond `2**53` and bounds near the float64 limit. Responses outside the bounds
raise `ValueError`, which catches mistyped bounds. The comparison is exact, so a
float16 or float32 response cannot pass a bound it equals only after rounding.
Missing responses stay `NaN`. Apart from converting non-array input, the returned
copy is the only full-size allocation: the selected columns are checked and
recoded in row batches of a few megabytes. Keep the responses as presented for
sequence and response-style indices such as `longstring`, `markov`, `irv`, `onset`, and
`acquiescence`, because recoding changes the response sequence they measure.
`screen()` and the composite helpers apply this split for you through
`IndexOptions.reverse_keyed_items`, described in
[Reverse-keyed items](#reverse-keyed-items).

`psychsyn` and `psychant` discover pairs from finite item correlations. Constant
items and samples with fewer than two respondents have undefined item
correlations and cannot supply pairs, even at a zero cutoff. The
`item_correlations` keyword (`IndexOptions.psychsyn_item_correlations`, CLI
`--psychsyn-item-correlations`) sets the missing-data policy for `psychsyn`,
`psychant`, `psychsyn_critval()`, and `psychsyn_summary()`:

- `"complete"` (the default) follows `np.corrcoef`: an item with any missing
  response is undefined. With scattered missingness nearly every item has a gap,
  so few or no pairs are found and scores are typically all `NaN`.
- `"pairwise"` correlates each item pair over the respondents who answered both,
  as `careless::psychsyn` does, and scores each respondent over the selected
  pairs they answered. Pairs need at least three shared respondents, because two
  shared responses always correlate perfectly; this also holds without missing
  responses, so fewer than three respondents yield no pairs. Otherwise, without
  missing responses both modes return identical results.

Use `"pairwise"` for data with missing responses. At least two selected pairs
(answered pairs in pairwise mode) are needed for a respondent score; otherwise
the score is `NaN` and remains unavailable in screening and composites.
Diagnostics count a single selected or answered pair. `resample_na` and
`random_seed` are accepted for compatibility with `careless::psychsyn` but have
no effect on scores: complete discovery selects only fully observed items,
pairwise mode leaves respondents with fewer than two answered pairs unavailable,
and a respondent whose answered pairs have zero within-pair variance scores 0.0
where careless returns `NA`. `na_rm` therefore does not change `psychsyn` or
`psychant` scores in `screen()` and `composite()`. Cutoffs must be finite real
numbers; `psychsyn_critval()` additionally requires a nonnegative minimum magnitude.
Item discovery preserves correlations at very large or small finite scales and
for nearly constant items with representable variation. In pairwise mode this
includes extreme or outlying responses outside a pair's shared respondents.
Constant decimal-valued items remain unavailable, and an infinite response
invalidates its item. Summaries with no available scores return `NaN` statistics
without warnings. `psychsyn_summary()` reports score statistics, the number of
selected `item_pairs`, and `n_total`, `n_valid`, and `n_missing` respondent counts.
Integer item discovery preserves small differences at large baselines. Person-total
profiles retain variation between nearly identical integer item means, while
Guttman scoring preserves exact integer difficulty rankings and response categories.

`psychsyn_flag()`, `psychant_flag()`, `person_total_flag()`, `u3_poly_flag()`, and
`midpoint_responding_flag()` return `(scores, flags)` in each index's registry
flag direction (low for `psychsyn` and `person_total`, high for `psychant` and
the response-style indices). Attentive respondents give strongly negative
antonym correlations, so `psychant` flags scores near zero or above.
Their default tail percentiles (5 for low, 95 for high) reproduce the per-index
flags of `screen()` at its default `percentile=95`. Fixed thresholds include
ties, percentile cutoffs exclude them, and unavailable scores are never flagged.

`guttman` orders items easiest first, by descending sample item mean, and keeps
items with tied means in column order. An error is an item pair where the
respondent scores strictly higher on the harder (less endorsed) item than on the
easier one, so a perfect cumulative pattern scores 0 and its reversal scores the
maximum. Item means always use each item's available responses, whatever
`na_rm` is, and items without any observed response are ordered last. Pairs
involving a missing response never count as errors, so raw counts do not depend
on `na_rm`; it only selects the normalization denominator. With `na_rm=True`,
normalized scores divide by the respondent's available pairs; with
`na_rm=False`, the denominator counts every item pair. `guttman_flag()` flags
normalized scores strictly above a finite `threshold` (default 0.5); like the
other flag helpers, it converts `threshold` with `float()`, so numeric strings
are accepted while `True` and `False` are rejected.

`mahad` flags only unusually large distances. `method="chi2"` compares squared
distances with the `confidence` quantile of a chi-squared distribution with one
degree of freedom per item, `"iqr"` uses the upper fence `Q3 + 1.5 * IQR`, and
`"zscore"` flags standardized distances above the two-sided normal critical value
for `confidence` (1.96 at 0.95). `confidence` must be a finite real number from 0
to 1: Python and NumPy scalars, 0-d arrays, `Decimal`, and `Fraction` values are
accepted, while Booleans and strings are rejected. `mahad_summary()` reports
distance statistics, the chi-squared outlier count, and `n_total`, `n_valid`, and
`n_missing` respondent counts.

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

The opt-in `avgstr` index scores the average length of uninterrupted identical
runs, `longstring_scores(x, avg=True)`: observed responses divided by the number
of runs, matching `mean(rle(x)$lengths)` from `careless::longstring(avg = TRUE)`.
Curran (2016) recommends it beside `longstring` because repeated short runs raise
the average even when no single run is long. Higher values are flagged.
Rows without observed responses are unavailable (`NaN`). With `na_rm=True`,
missing responses are removed before runs are counted, so a run can continue
across a skipped item; R's `rle()` instead ends a run at `NA`. `na_rm=False`
rejects missing responses.

`longstring_pattern()` and `IndexOptions.longstring_max_pattern_length` require an
integer `max_pattern_length` of at least 2, the shortest repeating pattern.
Booleans, fractional values, and smaller lengths raise `ValueError` instead of
silently scoring zero; the CLI rejects `--longstring-max-pattern-length` values
below 2.

`autocorrelation` (opt-in) correlates each respondent's responses with lagged
copies of themselves (Gottfried et al., 2022; `responsePatterns::rp.acors`). For
each lag `k` from `min_lag` (default 1) to `max_lag`, the responses `r[0:n-k]` are
correlated (Pearson) with `r[k:n]`. The score is the largest absolute lag
correlation (`statistic="max_abs"`, the default) or the sum of absolute lag
correlations (`"sum_abs"`), and higher values are flagged. Zigzags
(1-2-3-4-5-4-3-2), seesaws, and short cycles correlate strongly at their period
even when some responses are perturbed, which exact pattern matching
(`longstring_pattern`) misses and transition entropy (`markov`) tolerates less
well as noise grows. Items must be in presentation order and on one response
scale; keep reverse-worded items as presented.

The default `max_lag=10` deliberately departs from the R default of `J - 3`
lags: long lags pair few responses, and their noisy correlations dilute the
signal of short cycles. In a simulated 40-item survey with 1,000 attentive and
150 careless zigzag, seesaw, and cycle responders whose responses were 20%
randomly perturbed, `max_abs` separated the groups with an AUC of 0.96 at
`max_lag=10` but 0.75 at `J - 3`. Keep `max_lag` well below the item count on
short questionnaires; `max_lag=None` restores the R default for each respondent.

Each lag needs at least three response pairs, so a respondent's usable lags stop
at their number of observed responses minus 3, and respondents with fewer than
`min_lag + 3` observed responses are unavailable (`NaN`). Following the
zero-variance rule of `rp.acors`, a lag whose window `r[0:n-k]` or `r[k:n]` does
not vary scores 1, a perfectly repetitive pattern. Respondents whose responses
are all identical therefore score 1 at every usable lag, and a straight-liner
whose first or last response differs still scores 1 at every lag; a respondent
who answers a few items and then straight-lines scores 1 at the lags whose
second window falls within the run. Straight-lining thus overlaps with
`longstring`. With `na_rm=True`, missing responses are removed in sequence order;
`na_rm=False` leaves respondents with any missing response unavailable. Rows
containing infinite responses are unavailable. `return_lags=True` also returns
the lag with the largest absolute correlation, with ties going to the smallest
lag. Exact ties can differ by floating-point rounding, so correlations within a
relative `2**-42` of 1 score exactly 1 and lags within that tolerance of the
largest correlation count as tied: a perfect 1-2-3-4-5-4-3-2 zigzag reports lag
4, not 8. Registry scoring uses `IndexOptions.autocorrelation_max_lag` and
`IndexOptions.autocorrelation_statistic`, and the CLI accepts
`--autocorrelation-max-lag` (`none` for the R default) and
`--autocorrelation-statistic`. `autocorrelation_flag()` flags the high tail.
Window moments come from prefix sums of centered responses in bounded batches.
Nearly constant windows are recomputed from their own centered responses, so
very large offsets, tiny or huge response scales, and large integers keep
accurate correlations.

`gpoly`, `u3poly_fit`, and `ht` (all opt-in) are the nonparametric person-fit
statistics of the R package PerFit (Tendeiro et al., 2016), available directly as
`gpoly()`, `u3poly()`, and `ht()`. They ask whether a respondent's answers follow
the item ordering of the whole sample. Each item with `M + 1` ordered categories
is split into `M` item steps `X_j >= h`, and a step's popularity is the share of
respondents who pass it. Steps are ordered from most to least popular; ties keep
item order and then step order, like PerFit's `rank(ties.method = "first")`, so
the steps of one item stay nested. Niessen, Meijer, and Tendeiro (2016) found
polytomous Guttman errors effective for detecting careless respondents, and Ht
was among the best-performing statistics in Karabatsos (2003).

- `gpoly()` counts polytomous Guttman errors (Molenaar, 1991): pairs of steps
  where the respondent fails the more popular step but passes the less popular
  one, PerFit's `Gpoly`. `normalize=True` (the default, and the registry score)
  divides by the largest count any response pattern with the same total score
  can reach (Emons, 2008), PerFit's `Gnormed.poly`; for dichotomous items this
  is `G / (r * (J - r))`. Unlike `guttman(normalize=True)`, which divides by every
  item pair, the normalized value is not confounded with the total score.
  Counts and maxima are exact integers. Each maximum passes every step of some
  items and part of at most one more, so all maxima take `O(J * M)` work.
- `u3poly()` weights each step by the log-odds of its popularity and scores
  `(W_max - W) / (W_max - W_min)`, where `W` sums the weights of the passed steps
  and `W_max` and `W_min` are its extremes over all nested patterns with the same
  total score (van der Flier, 1982; Emons, 2008), PerFit's `U3poly`. Scores run
  from 0 for a perfect Guttman pattern to 1. A step passed by every respondent, or
  by none, is weighted as if half a response went the other way. Weights share
  one integer fixed-point scale, so the extremes, perfect patterns, and the
  lowest-weight patterns are exact.
- `ht()` is the transposed scalability coefficient (Sijtsma & Meijer, 1992),
  PerFit's `Ht`, for responses coded 0 and 1; other values raise `ValueError`.
  It divides the summed covariances, across items, between a respondent's
  answers and everyone else's by the sum of their maxima given each pair's
  numbers of correct answers. Lower values are flagged. It is computed in exact
  integers in `O(N * J)` time, without the `N`-by-`N` covariance matrix.

Responses must be integer categories on one scale shared by all items. The
scale runs from `scale_min` to `scale_max` (`IndexOptions.scale_min` /
`scale_max`), inferred from the observed minimum and maximum when omitted, and
`ncat` (`IndexOptions.person_fit_ncat`, CLI `--person-fit-ncat`) sets the number
of categories from either endpoint, or from the observed minimum when neither is
given. Pass `scale_min` when the lowest category may be unobserved, since PerFit
codes categories from 0. Fractional responses, including extended-precision
values within rounding of an integer, responses outside the scale, and scales
with more than 1024 categories raise `ValueError`. Like `guttman`, these
statistics assume every item is keyed in one direction; recode reverse-worded
items with `reverse_score()` first, or set `IndexOptions.reverse_keyed_items` so
the registry recodes them for these indices. With `person_fit_ncat` set, the
registry reflects the items for `gpoly` and `u3poly_fit` on the scale that `ncat`
declares, so a category nobody chose still sets the reflection. `ht` and the
other keyed indices keep `scale_min` and `scale_max`, inferred when omitted, and
read a second recoded copy unless both endpoints are given.

Scores are `NaN` when the total score allows no misfit: every step failed or
passed for `gpoly` and `u3poly`, and every or no item answered correctly for
`ht`. PerFit's `Gnormed.poly` and `U3poly` score those all-minimum and
all-maximum response vectors 0 instead. Step popularities use each item's
available responses. With `na_rm=True`, raw `gpoly(normalize=False)` counts only
pairs of observed steps, while normalized `gpoly`, `u3poly`, and `ht` leave
respondents with missing responses unavailable (`NaN`); `ht` also leaves them out
of everyone else's comparisons. `na_rm=False` raises `ValueError` for missing
responses. PerFit's default `NA.method = "Pairwise"` instead scores incomplete
respondents from their observed items, and only its `Hotdeck`, `NPModel`, and
`PModel` methods impute missing responses, so incomplete respondents differ.
`gpoly_flag()` and `u3poly_flag()` flag the high tail and `ht_flag()` the low
tail (default 5th percentile).

In five simulated samples of 1,000 graded-response-model respondents and 100
uniform random responders on 30 five-category items, normalized `gpoly`
separated the groups with AUCs of 0.93–0.99, raw counts 0.97–0.99, and `u3poly`
0.92–0.98. On 20 two-parameter logistic items, `ht` reached 0.98–1.00, beside
0.98–1.00 for `lz`.

`u3_poly` is unrelated to PerFit's U3 despite its name: it is the proportion of
extreme (endpoint) responses, a response-style index. Use `u3poly()` or the
registry's `u3poly_fit` for the person-fit statistic.

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
When row variation approaches underflow, bounded power-of-two scaling preserves
the ratio of paired differences to standard deviation, including inferred
antonym bounds. Constant and unavailable rows retain their existing behavior.

`missing_rate` is opt-in because planned skip logic and matrix preprocessing can
create legitimate omissions. Use `IndexOptions.missing_item_indices` to restrict
registry scoring to a fixed required-item subset, or pass the same subset as
`item_indices` to the standalone helper. For respondent-specific skip logic,
provide a Boolean `missing_applicable_mask` through `IndexOptions` or
`applicable_mask` directly. False cells are excluded from both the numerator and
denominator; rows without applicable selected items return `NaN` and are not
flagged. The CLI exposes fixed subsets through `--missing-item-indices` and
respondent-specific masks through `--missing-applicable-mask PATH`.

`infrequency` preserves its historical missing-response behavior with
`missing="pass"`: unanswered checks do not count as failures and remain in a
proportion denominator. Choose `"fail"` for conservative scoring, `"omit"` for
available-case proportions, or `"propagate"` to require complete attention-check
data. Configure registry scoring with `IndexOptions.infrequency_missing` and the
CLI with `--infrequency-missing`. The standalone `infrequency_flag()` can flag
either counts or proportions.

Instructed items ("select Strongly Agree") have one correct answer, supplied
through `expected_responses`. Bogus and infrequency items (Meade & Craig, 2012)
usually accept several answers: both disagreement categories
are correct for "I have been to every country in the world", and only agreement
indicates inattention. Self-reported diligence or "use my data" items likewise
accept a band of ratings. Supply one inclusive `(low, high)` range per item
through keyword-only `acceptable_ranges` instead, with `-inf` or `inf` for an
open end:

```python
from ier import infrequency

# Item 3: bogus item on a 1-5 scale; disagreement (1-2) is correct.
# Item 7: diligence self-report on a 1-7 scale; ratings of 5 or more pass.
failures = infrequency(
    responses, item_indices=[3, 7], acceptable_ranges=[(1, 2), (5, float("inf"))]
)
```

Exactly one of `expected_responses` or `acceptable_ranges` must be given. Range
bounds must be real numbers with `low <= high`; `NaN` bounds are rejected.
Integer responses compare against the integer categories inside each range, so
fractional bounds round inward and large integer bounds stay exact; a range
containing no category fails every observed response. Single-precision responses
are compared with double-precision bounds. Missing responses follow the
configured missing policy. Registry scoring uses
`IndexOptions.infrequency_acceptable_ranges`, which can replace
`infrequency_expected_responses` (the catalog lists the two as an
`alternative_options` group), and the CLI accepts
`--infrequency-acceptable-ranges '1:2,5:'`, with an empty side for an open end.
The two answer options cannot be combined. The command line can mistake a
separate argument that starts with `-` for another option, so write an open
lower end as `:-1` and attach values that begin with a minus sign with `=`:
`--infrequency-acceptable-ranges=-3:-1` or
`--infrequency-expected-responses=-1,2`.

## Reverse-keyed items

Consistency and person-fit indices assume that every item is keyed in the same
direction, so reverse-worded items must be recoded before scoring. Sequence and
response-style indices measure the responses as presented, and recoding would
distort them: after recoding, a straight-liner no longer gives identical answers
and a respondent who agrees with every item no longer looks acquiescent. In a
simulated survey of six eight-item scales with a third of the items
reverse-worded, the classic even-odd index (`method="halves"`) separated 100
uniform random responders from 1,000 attentive respondents with an AUC of 0.62
on raw data and 0.96 on recoded data, and scale-aware `individual_reliability`
with 0.54 and 0.99. Against 100 straight-liners, even-odd was unavailable for
every straight-liner on raw data and reached an AUC of 1.00 on recoded data,
while `longstring` reached 1.00 on raw data and 0.17 on recoded data. No single
matrix serves both groups.

Set `IndexOptions.reverse_keyed_items` to the 0-based columns of the
reverse-worded items, and `screen()`, `composite()`, and the other composite
helpers recode them once with `reverse_score()` for the indices that need keyed
responses, while every other index reads the matrix as given:

```python
from ier import IndexOptions, screen

options = IndexOptions(
    scale_min=1,
    scale_max=5,
    evenodd_factors=[8, 8, 8],
    evenodd_method="halves",
    reverse_keyed_items=[1, 4, 7, 9, 12, 15, 17, 20, 23],
)
result = screen(data, indices=["evenodd", "guttman", "longstring", "irv"], options=options)
```

The CLI accepts the same list with `--reverse-keyed-items 1,4,7` on `ier screen`
and `ier composite`, and a configuration file accepts
`reverse_keyed_items = [1, 4, 7]`.

| Responses | Indices | Reason |
|-----------|---------|--------|
| Recoded | `evenodd`, `individual_reliability` | Split halves of a scale agree only when all of its items point the same way |
| Recoded | `guttman`, `lz`, `gpoly`, `u3poly_fit`, `ht` | Cumulative, item-step, and item response models assume every item increases with one trait |
| As presented | `irv`, `longstring`, `longstring_pattern`, `avgstr`, `markov`, `autocorrelation`, `onset` | They measure the response sequence as given |
| As presented | `acquiescence`, `u3_poly`, `midpoint` | Agreement uses raw responses; extreme and midpoint categories are unchanged by reflection |
| As presented | `mad`, `semantic_ant` | Each pair already reverse-scores its second item |
| As presented | `psychsyn`, `psychant`, `semantic_syn` | Pairs are discovered or configured on the presented items; antonym pairs come from reverse-worded items |
| As presented | `person_total` | The reference profile is the sample's own item means |
| As presented | `mahad` | Reverse scoring is affine per item, so distances do not change |
| As presented | `infrequency`, `missing_rate` | Expected answers refer to presented responses, and recoding preserves missingness |

`index_catalog()` and `ier indices` report the same split as
`uses_keyed_responses`. Bounds come from `scale_min` and `scale_max` and are
inferred from the whole matrix when omitted, so pass them whenever an endpoint
might be unobserved. Items must be distinct in-range columns and every recoded
response must lie within the bounds. Otherwise each selected keyed index fails
with a `reverse_keyed_items could not be applied` message, recorded in the
result's errors like other configuration failures or raised with `strict=True`,
while the other indices are still scored. The recoded copy is made once per run
and shared by the keyed indices, including with `workers` above 1, and
`reverse_score()` works through the selected columns in row batches of a few
megabytes, so peak memory grows by about one recoded matrix only when the option
is set and a keyed index is selected. A second copy is made only when
`person_fit_ncat` is set without both scale bounds and `gpoly` or `u3poly_fit`
is selected alongside another keyed index, because the person-fit indices then
reflect items on their declared category scale. That matrix has the dtype `reverse_score()`
returns: float64 for floating responses, twice the size of a float32 input, and
usually the input dtype for integer responses. On a 50,000 × 60 float64 survey
(24 MB) with a third of the items reversed, `screen()` with `evenodd` peaked
24 MB higher with the option than without it. Without a keyed index the option
is ignored, like other index-specific options.

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
| `response_time_effort` | Share of answered items at or above item thresholds (RTE) | low (rapid responding) |
| `response_time_effort_flag` | Fixed RTE cutoff flagging (default below 0.90) | low |

Response-time medians omit missing items and return `NaN` for entirely missing
respondents. Middle pairs preserve finite medians at very large or small scales;
`float32` pairs are averaged in double precision and integer pairs are averaged
before rounding the result to a float.

Response-time coefficients of variation preserve dimensionless ratios even for
varying subnormal observations, avoiding a zero or undefined result caused only
by separately rounding the mean and standard deviation. Positive constant rows
still have zero variation; zero means and unavailable observations retain their
documented arithmetic behavior.

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

`response_time_effort()` computes response time effort (RTE; Wise & Kong, 2005):
the proportion of a respondent's answered items whose response time is at or
above that item's threshold. The person-level summaries above reduce each row to
one number first, so item length and difficulty confound them; RTE judges each
response against its own item, so a respondent who skims long items while
answering short ones at a normal pace still loses effort. Without explicit
`thresholds`, every item uses the normative NT10 threshold (Wise & Ma, 2012):
`normative_fraction` (default 0.10) times the item's exact mean time over the
respondents who answered it. `max_threshold` caps every threshold, for example
at the 10 seconds Wise and Ma recommend. Pass `thresholds` as one shared value or
one value per item, in the units of the timing matrix, to apply another rule.

A response is rapid when its time is strictly below the threshold. Missing
times are unanswered items. Items whose threshold, after any `max_threshold`
cap, is missing, infinite, or not positive, including items nobody answered, are
excluded from every respondent's denominator, and respondents without an
answered eligible item are unavailable (`NaN`). This applies to explicit
`thresholds` as well as normative ones, so `NaN`, `inf`, and `-inf` thresholds
exclude their items, except that `max_threshold` caps `inf` like any other
threshold. Integer and `float32` times are compared with double-precision
thresholds. `return_item_flags=True` also returns the Boolean matrix of rapid
responses for effort-moderated scoring. `response_time_effort_flag()` flags RTE
strictly below `threshold` (default 0.90) and never flags unavailable
respondents. Like the other timing helpers, these functions are not registry
indices.

```python
from ier import response_time_effort, response_time_effort_flag

effort, rapid = response_time_effort(times, max_threshold=10.0, return_item_flags=True)
scores, low_effort = response_time_effort_flag(times, threshold=0.90, max_threshold=10.0)
```

`ier response-time --metric effort` scores RTE from a timing file.
`--effort-fraction` (default 0.10) and `--effort-max-threshold` set the
normative thresholds; `--effort-threshold` instead applies one shared item
threshold, in the timing units, and cannot be combined with either.
Effort follows Wise and Kong's absolute rule rather than a sample percentile:
without a cutoff option it flags RTE strictly below 0.90, matching
`response_time_effort_flag()`. RTE is a proportion of answered items, so many
respondents share exact values such as 1 or 9/10; a sample percentile would move
with those ties and with how many respondents rushed, while a fixed cutoff keeps
one meaning across samples. `--threshold` replaces the cutoff with another RTE
value between 0 and 1, still flagged strictly below, and `--percentile` applies
a tie-exclusive sample cutoff. NPZ output records the metric and its decision,
and `ier response-time-scores` reflags saved effort scores with the same strict
rule.

```bash
ier response-time timings.csv --metric effort --effort-max-threshold 10
ier response-time timings.csv --metric effort --effort-threshold 2 --threshold 0.8
ier response-time timings.csv --metric effort --format npz --output effort.npz
ier response-time-scores effort.npz --threshold 0.95 --format csv
```

## Persisting screening results

`save_screen_archive()` stores a complete `screen()` or `screen_scores()` result
in the same versioned, pickle-free NPZ schema as `ier screen --format npz`.
`load_screen_archive()` restores the `ScreenResult` from either source, so a
saved run can be inspected or plotted without the original responses:

```python
from ier import load_screen_archive, plot_flag_counts, save_screen_archive, screen

result = screen(data, indices=["irv", "longstring", "onset"], percentile=95)
save_screen_archive("screening.npz", result, respondent_ids=ids, compressed=True)

saved = load_screen_archive("screening.npz")
restored = saved["result"]
print(restored["threshold_sources"], saved["respondent_ids"][:3])
figure = plot_flag_counts(restored)
```

The loader recomputes each index's flags from its scores and recorded cutoff
(fixed thresholds include ties, percentile cutoffs exclude them, presence rules
flag available scores) and rejects archives whose flags, counts, consensus
decisions, summary counts, flag rates, or score minima and maxima disagree.
Summary means and standard deviations are rebuilt from the restored scores, so
a loaded summary always describes the loaded scores. Reload the archive to keep
the original decisions: passing `restored["thresholds"]` back to
`screen_scores(thresholds=...)` turns percentile cutoffs into inclusive fixed
thresholds and flags tied scores.
Use `load_score_archive()` when only the raw scores are needed for new cutoffs.

## Plot helpers

Requires `insufficient-effort[plot]`:

- `plot_distributions(screen_result)`
- `plot_flag_counts(screen_result)`
- `plot_flagged_heatmap(screen_result)`
- `plot_index_agreement(screen_result, kind="jaccard")`
- `plot_composite(scores, threshold=None, flags=None)`
- `mahad_qqplot(...)`

Distribution panels draw each applied cutoff as a dashed line and shade the tail
that index flags: left of the cutoff for low-direction indices and right of it
for high-direction indices. Panel titles report flagged counts. Presence-flagged
onset and indices without any available scores have no cutoff line, and
unregistered score names get a line without shading. Pass
`show_thresholds=False` for plain histograms.

Heatmaps use a fixed flag color scale, so fully flagged and wholly unflagged
cohorts remain visually distinct. Unavailable scores appear gray with a legend.
For onset, an absent detection remains unflagged because NaN represents no
detected event. Automatic heatmap dimensions are capped at 18 × 12 inches; pass
`figsize` to choose different dimensions. A figure has only about a thousand
pixel rows, so drawing every respondent of a large survey would silently drop
most rare flags. Surveys with more than `max_rows=1000` respondents are instead
drawn as consecutive respondent blocks of `ceil(n_respondents / max_rows)`. A
block is flagged when any of its respondents is flagged and gray only when all
of them are unavailable, so a single flag among 100,000 respondents stays
visible. Pass `max_rows=None` for one row per respondent, or `order="flag_count"`
to sort respondents by descending flag count first so flagged respondents
cluster at the top. In a local Agg comparison, a 1,000,000-respondent,
11-index heatmap rendered in 0.15–0.42 s instead of 1.7–2.2 s, and showed
610–660 flagged pixel rows per index instead of 0–4. Flag-count charts aggregate
respondents in a single pass and retain bins for every selected index.

`plot_index_agreement()` draws `index_agreement()` with fixed color limits:
`[0, 1]` (`viridis`) for Jaccard similarity and for overlap, shown as the share
of respondents flagged by both indices, and `[-1, 1]` (`RdBu_r`) for Spearman
correlations. Spearman scores are oriented by flag direction, so red cells mean
both indices rank the same respondents as more suspicious and blue cells mean
they disagree. Undefined cells are gray. `plot_composite()` draws a composite
score histogram with an optional cutoff and shaded high tail, matching
`composite_flag()`. Its legend counts supplied flags, or scores at or above the
cutoff when no flags are given.
