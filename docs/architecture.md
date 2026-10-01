# Architecture

This note explains how IER orchestrates indices, handles missing data, and
flags respondents. It is aimed at contributors and methods-curious users.

## Layers

```text
CLI / examples
      │
screen() / composite()     ← public orchestration
screen_scores() / composite_scores() / response_time_score_flags()
                                    ← reusable decision layer
      │
IndexOptions + registry    ← shared config + index catalog
      │
per-index modules          ← irv, longstring, mahad, …
      │
_validation / _flagging    ← shared input checks and threshold helpers
```

- **NumPy-first core.** Base installs depend only on NumPy. Statistical
  routines, including chi-square flagging, response-time mixtures, and IRT
  theta estimation, are implemented locally. Matplotlib is only required for
  plotting helpers.
- **Registry.** `src/ier/_registry.py` maps string names to scorers, default
  screen/composite membership, flag direction, and required `IndexOptions`
  fields (e.g. `evenodd_factors`).
- **Soft per-index errors.** `score_registered_indices()` catches validation /
  runtime failures per index and returns them in an `errors` dict instead of
  aborting the whole screen. Public orchestration APIs also accept `strict=True`
  to raise a contextual error as soon as a selected index fails. Composite
  callers still raise if *no* index succeeds under the default soft policy.
- **Opt-in parallelism.** Registry orchestration evaluates indices sequentially
  by default. `workers>1` uses a lazily imported standard-library thread pool,
  then records scores and failures in selection order. This keeps default import
  cost and resource use stable while allowing NumPy-heavy scorers to overlap.
- **Bounded reductions.** Screening flag counts, valid-score counts, and summary
  coverage share one index-at-a-time pass; composite mean, sum, and maximum values
  use the same bounded approach. Orchestration retains its documented per-index
  result vectors but does not construct another complete respondent-by-index
  matrix for final reductions.
- **Composite reductions.** Component calibration and all three reduction methods
  share `_composite_reductions`. Ordinary weighted means and sums normalize safe
  common weight units. Instability triggers bounded respondent batches that scale
  numerator products and, for means, available weight totals independently.
  Maximum repairs compare product signs, exponents, and mantissas before
  converting the winner, allowing oversized negative contributions to be
  discarded. Exact scalar arithmetic repairs severe cancellation and subnormal
  rounding. Final sums and maxima outside the finite float range raise a
  contextual error; respondents below the requested coverage minimum retain
  `NaN`. Calibrations retain only scalar metadata between components, allowing
  repair without another complete respondent-by-index matrix. Single-component
  reductions own their output buffer, and means omit the cancelling weight.
  Unweighted mean summaries reuse coverage counts as their denominator.
  Ordinary component calibration also returns its availability mask, using a
  scalar complete-coverage marker to enable unmasked accumulation and scalar
  count increments. Sum presence vectors are released once a complete component
  supplies every respondent. Cancellation checks use boolean comparisons rather
  than a full absolute-value float buffer.
- **Score statistics.** Shared summaries reuse stable row means, deviations, and
  medians on observed score vectors. Screening and composite summaries reuse
  complete vectors directly. Composite standardization owns a scaled observation
  buffer, shifting nearby values before centering to preserve small differences
  from a common baseline and leaving caller-owned arrays untouched.
- **Paired-item validation.** Balanced acquiescence and MAD share one ordered-pair
  boundary for integer conversion, matrix bounds, and equal cardinality. Their
  bounded numeric kernels therefore receive exact pairs and never discard a
  trailing configured item.
- **Correlation workspaces.** Row correlations used by even–odd, person-total,
  psychometric consistency, and individual reliability scoring use bounded
  respondent batches. Two
  paired observations use their relative order directly; longer vectors use
  centered reductions with separate norm divisions and rescaling for extreme
  magnitudes. Missing pairs and zero-variance policies remain distinct.
  Complete person-total batches reuse a single centered item-mean profile and
  its norm instead of allocating the same profile for every respondent.
- **Reusable decisions.** `screen_scores()` reapplies direction-aware flagging,
  completeness, consensus, and summaries to retained registered score vectors.
  `composite_scores()` reuses raw composite-enabled vectors for alternative
  weights, standardization, completeness, and reductions.
  `response_time_score_flags()` reapplies low- or high-tail cutoffs to retained
  direct timing scores or mixture probabilities. All three support sensitivity
  analysis without rerunning scorers.

## Command-line boundaries

The command-line path is split by responsibility:

- `cli.py` defines arguments, converts index options, and coordinates commands.
- `_cli_streams.py` owns suffix-selected, standard-library text compression.
- `_cli_input.py` owns forward-only delimited input, preamble removal,
  named-column selection, and memory-mapped NumPy input.
- `_cli_composite.py` validates shared respondent alignment and flag metadata
  contracts for every composite serializer.
- `_cli_output.py` renders text plus bounded strict JSON and CSV results.
- `_cli_npz.py` assembles complete command result payloads and delegates the
  low-level typed, pickle-free NumPy writer.
- `_atomic_output.py` stages regular result files for replacement after every
  stream closes, sharing permission and symbolic-link handling between text and
  archive writers. Special-file destinations retain direct streaming.
- `archive.py` owns the shared archive stream writer and
  public validated save/load boundaries for reusable registered score vectors
  and response-time results.

Screen and composite commands carry the registry's ordered soft-failure map
through text, JSON, and NPZ serializers and mirror failures to standard error
for every format. Score computation still runs once, and CSV remains a compact
respondent table.

Detailed composite output is explicit through `--include-components`. The CLI
reuses `composite_summary()` so scoring still runs once, JSON wraps each
component in the bounded array writer, CSV emits one row at a time, and NPZ
writes separate typed members without stacking another respondent-by-index
matrix. The default aggregate-only path does not allocate or serialize these
details.

Composite standardization is resolved before either aggregate-only or detailed
scoring begins. The same boolean is passed to the public scoring API and written
to text, JSON, and NPZ metadata, avoiding a second calculation or inference from
the resulting values. CSV intentionally remains a respondent-only table.

Optional composite flagging runs after aggregate scoring and reuses that score
vector. Cutoff resolution and boolean comparison use the same shared helpers as
the public flagging APIs. JSON writes the flag vector in bounded chunks, CSV
streams it row by row, and NPZ stores it as a boolean member; no component is
rescored and the score-only path allocates no flag vector.

Keeping parsing, matrix construction, serialization, and orchestration separate
makes format-specific changes independently testable without adding runtime
packages or coupling them to statistical index implementations.

## IndexOptions

Orchestration APIs accept configuration **only** via
`options=IndexOptions(...)`. That keeps `screen()` / `composite()` signatures
stable as indices grow. Per-function kwargs remain available when calling an
index module directly (e.g. `mahad(x, method="iqr")`).

## Flagging policy

`screen()` turns scores into boolean flags using registry metadata:

| Mode | Typical indices | Rule |
|------|-----------------|------|
| Low / high percentile | IRV, longstring, mahad, … | Extreme tail relative to the sample |
| Fixed low / high threshold | Any percentile-mode index | At or beyond a validated cutoff |
| Presence | `onset` | Any detected changepoint is flagged |

Percentile defaults are **sample-relative heuristics**, not calibrated
diagnostic cutoffs. `screen(thresholds=...)` accepts fixed cutoffs when a survey
or validation study provides them, while `screen(percentiles=...)` tunes
sample-relative tail sensitivity by index. Results retain the actual cutoff,
its fixed/percentile/presence source, and the requested tail percentile. See
[Threshold Guidance](thresholds.md).

Sample cutoffs use linear interpolation at rank `(n - 1) * percentile / 100`
among available scores. The shared selector partitions an owned observation
buffer without reordering retained scores. Interpolation uses double precision,
with exact scalar repairs for overflowing spans, subnormal differences, and
integer endpoints. Percentiles 0 and 100 use the observed minimum and maximum;
an entirely unavailable sample retains cutoff zero and produces no flags.
Integer score comparisons preserve exact values, and single-precision arrays
compare against the double-precision cutoff without narrowing it.

## Missing data

Most indices honor `IndexOptions.na_rm` (default `True`) and document
row-wise vs listwise behavior in their docstrings. Policies are intentionally
not identical across indices: IRV can use `nanstd`, Mahalanobis may drop
incomplete rows, Markov may require complete sequences depending on `na_rm`.
When comparing to R packages, align NA policy first.

The opt-in `missing_rate` index separates observed missingness from survey
applicability. A fixed `missing_item_indices` subset works through registry and
command-line scoring. Python callers can additionally provide a Boolean
respondent-by-item applicability mask; its false cells do not contribute to
either the missing count or denominator. A zero denominator yields `NaN`, which
the shared flagging policy leaves unflagged.

Attention-check scoring keeps its legacy missing-as-pass behavior by default but
also supports missing-as-failure, available-case omission, and strict propagation.
The policy is carried through `IndexOptions`; unavailable scores reuse the same
flagging and composite-coverage rules as every other registry index.
Missing-response and attention-check scoring share ordered item-selection
validation. Both select columns and reduce responses in bounded respondent
batches; missing-response scoring selects applicability cells within the same
batch instead of copying the entire selected mask. Attention checks compare
selected items together and write directly into a single result vector.

Screening counts available scores alongside flags without stacking either set of
vectors. An optional `min_valid_indices` rule marks rows with insufficient score
coverage as ineligible for consensus while retaining the individual scores,
flags, counts, and eligibility decision for audit.

## Composite scores

`composite()` z-combines selected indices with direction multipliers so that
higher composite values mean more evidence of careless responding.
Optional positive weights are applied after direction correction and
standardization. Weighted means renormalize over available scores per
respondent, so an unavailable index does not silently dilute the remaining
evidence.
Direction multipliers and weights are applied one vector at a time during the
reduction. Raw component vectors remain available for detailed summaries, while
ordinary and reusable composite paths avoid retaining a second mapping of
direction-corrected arrays.
An optional minimum valid-index rule masks under-supported respondent scores
after reduction. Equal-weight means reuse their existing denominator counts, so
the rule needs no additional respondent-sized workspace on that path; other
methods allocate one integer count vector only when the rule is enabled.
`composite_probability()` applies a logistic transform for convenience — it is
**not** a calibrated probability of carelessness. Do not treat it as a
posterior or diagnostic probability without your own validation study. Its
shared piecewise NumPy kernel evaluates positive and negative values separately,
avoiding overflow while preserving finite-tail precision and exact infinite
endpoints. The lz theta and likelihood paths reuse the same kernel for both
complete and missing-response batches. Each bounded batch excludes missing
items from the ability score equation, information, and likelihood reductions
when `na_rm=True`; entirely missing rows remain unavailable. With `na_rm=False`,
missing responses propagate through ability estimation. This shares one solver
and likelihood implementation across both input paths without per-respondent
Python loops.
Binary response matrices are reused without modification; polytomous inputs
allocate a converted matrix and fill it in bounded blocks. Difficulty estimation
shares the bounded item-mean reduction used by Guttman and person-total scoring,
and discrimination estimation uses bounded row totals. Discrimination estimates
reduce item means, item-specific total-score means, and centered cross-products
in bounded respondent batches, replacing per-item sorting and correlation
matrices. The binary-item variance uses `n * p * (1 - p)`; total-score variance
uses centered observations to preserve constant-score fallbacks even when
different items have different observed respondents. Both centered buffers
share the element budget. These reductions avoid full-matrix NaN replacement
copies. Items with no observed responses have unavailable difficulty estimates
and are omitted when `na_rm=True`.
The CLI computes this transform from the final aggregate vector only when
`--include-probability` is requested. JSON and CSV then serialize it
forward-only, while NPZ stores one additional typed vector; index scoring is not
repeated and default output schemas remain unchanged.

## Response-time helpers

Timing matrices are a different data modality than item responses. Helpers in
`response_time.py` are public but **intentionally outside** the screen /
composite registry so they are not mixed into item-response pipelines by
accident. Gaussian-mixture fitting reuses responsibility and scratch buffers
throughout EM, with contiguous component columns and a separate returned probability
vector so other component buffers can be released. Its expectation step follows a
fast probability-space path for ordinary observations and normalizes subnormal or
overflowed densities in log space. Very narrow components also use log densities
so their large normalization factor cannot amplify an underflowed exponential.
Only unrepresentable or nearly tied extreme log
densities need Decimal arithmetic. Large timing magnitudes are scaled by powers
of two, nearly constant data are shifted before fitting, and component standard
deviations retain the variance floor in the original fitting units. Weighted
deviations rescale exceptional residuals before squaring. Empty components receive
zero weight and cannot supply the fast component; constant medians return equal
component probabilities without iterating EM.
Retained summary vectors and mixture probabilities pass through the same shared
single-vector validation and threshold boundary, so cutoff sensitivity analysis
does not recalculate row summaries or refit EM.

Markov transition entropy discovers and encodes categories once per bounded
sequence block, reusing the encoded values for transition counting. Dense tables
contain only the block's observed states, up to 64; higher-cardinality blocks
use each row's observed states and pairs. Integral labels retain their original
precision, including adjacent 64-bit integers beyond floating-point precision.
Both paths evaluate the equivalent count form of conditional entropy.

Longstring and repeating-pattern scoring share bounded sequence preparation for
complete and missing-response inputs. Batches with missing responses compact
observed values in their original order; NaN padding and observed lengths exclude
artificial runs and overlong candidate patterns. Both indices reuse a cumulative
run-length kernel that counts consecutive matches without per-column Python loops
or scalar fallbacks. Markov scoring reuses the same sequence preparation, masks
padded transitions, and retains its sparse fallback for high-cardinality responses.
All-missing rows still return zero for longstring indices and NaN for Markov
entropy, and `na_rm=False` still rejects missing responses. These paths use NumPy
kernels without requiring a native extension or compiler.

For a local benchmark against commit `7143d28`, a 100,000-by-80 complete-response
matrix produced the following median times (five runs after one warmup, seed
20260927, Python 3.14.7, NumPy 2.3.5, `OPENBLAS_NUM_THREADS=1`):

| Operation | Before | Cumulative runs | Speedup |
|-----------|-------:|----------------:|--------:|
| Longstring | 73.3 ms | 27.6 ms | 2.7× |
| Repeating patterns | 494.3 ms | 143.0 ms | 3.5× |
| Default screening | 1144.2 ms | 652.3 ms | 1.8× |
| Default composite | 282.7 ms | 200.0 ms | 1.4× |

Reproduce this workload with `benchmarks/bench_sequence_scoring.py --respondents
100000 --missing-rate 0`; results depend on the machine and response distribution.
Peak traced allocation fell from 22.89 to 2.07 MiB for repeating patterns and
from 3.15 to 1.54 MiB for longstring. Full screening remained at 82.45 MiB because
other indices determine its peak. Smaller inputs can use more temporary memory:
complete 10,000-by-80 longstring scoring increased from 0.32 to 0.85 MiB. The
shared row budget bounds these workspaces independently of respondent count.

For the Markov refactor against commit `1e76814`, the same 100,000-by-80 workload
gave these medians with five alternating before/after runs following one warmup
per implementation (same seed, Python, NumPy, and BLAS settings as above):

| Operation | Missing responses | Before | After | Peak allocation before → after |
|-----------|------------------:|-------:|------:|-------------------------------:|
| Markov | 0% | 130.9 ms | 66.0 ms | 77.86 → 5.41 MiB |
| Markov | 10% | 153.1 ms | 135.3 ms | 9.22 → 9.22 MiB |
| Default screening | 0% | 789.0 ms | 725.0 ms | 82.45 → 70.39 MiB |

Use `benchmarks/bench_sequence_scoring.py --respondents 100000 --missing-rate 0
--operations markov screen` to reproduce the complete-response workload, and
`--missing-rate 0.1` for omissions. Whole-workflow gains depend on the other
indices: missing-data screening timing varied between runs, with this alternating
comparison measuring 876.4 ms before and 919.2 ms after and unchanged 63.40 MiB
peak allocation. Small response scales benefit most; 64- and 65-state inputs
showed little timing change. Allocation tracing remains separate from timing.

Even–odd consistency reduces factor correlations directly into respondent-level
sums and valid-factor counts. Correlation kernels use centered row workspaces and
contraction reductions, so peak allocation does not grow with the factor count.
Split IRV similarly accumulates section deviations inside bounded respondent
blocks without retaining a score vector for every section. Row-contiguous inputs
reduce adjacent equal-width sections together through array views; column-contiguous
inputs reduce one section at a time. Both reuse the last-axis mean/deviation kernel,
and singleton sections require only finiteness checks. Empty automatic sections are
excluded before allocation, and standalone row deviations no longer retain an
unused full-length mean vector.
Psychometric synonym and antonym scoring reuse the same row-correlation kernel
in bounded respondent batches, centering owned pair-selection buffers in place.
Complete and missing inputs share this path. Finiteness checks inspect selected
items once per block, and rows with unavailable selected responses skip correlation
work. Missing responses therefore do not trigger a complete respondent-by-pair
contribution matrix, and seeded resampling is also reduced in bounded chunks.
Undefined item correlations never become candidate pairs; fewer than two selected
pairs leave respondent scores unavailable. Common summary reductions filter missing
scores once and return unavailable statistics when no observed scores remain.
Item correlations for synonym, antonym, and cutoff discovery share a centered
cross-product reduction over bounded row blocks. Items with unavailable means
are excluded from multiplication and restored as undefined correlations; this
preserves column-wise missing-value propagation rather than introducing
pairwise deletion. Constant items and samples with fewer than two respondents
also have undefined correlations. Centering no longer copies the full response
matrix, while the item-by-item output still needs quadratic space in item count.
Cross-products are normalized directly, avoiding the covariance divisor that
cancels in a correlation. Constant candidates are verified against their original
observations and left undefined without repeating the full matrix calculation.
Other overflow, underflow, and variance very small relative to an item's mean
trigger bounded power-of-two rescaling. That path subtracts a sample observation
before accumulating the mean, preserving small differences. Real and complex inputs
share this policy; complex component scales avoid overflowing their magnitude.
Shared item means also accumulate in double precision, skip masked reductions for
complete blocks, and repair overflowing finite means using bounded rescaling. This keeps
person-total profiles and Guttman item ordering available at large finite scales.
Integer item means use native totals when their range permits; exceptional means
accumulate high and low integer parts within each respondent block. Person-total
profiles subtract a common exact total before final conversion and division,
retaining differences that would disappear in raw floating-point means. Guttman
orders exact integer totals and retains integer response categories. Integer item
correlations subtract each item's exact minimum before floating-point scaling,
so discovery preserves small differences on large baselines.
Predefined semantic pairs and MAD item pairs share a bounded absolute-difference
reducer. Pair selection, optional reverse scoring, and missing-aware means stay
within the common element budget instead of materializing complete pair matrices.
Exceptional reverse scoring subtracts each scale endpoint separately, avoiding
overflow and cancellation in their sum. Overflowing differences recover their
original responses and reduce in power-of-two-scaled units. Semantic consistency
batches row deviations alongside those pair reductions and can divide before
restoring units, preserving finite ratios when the raw difference mean exceeds
float range. MAD restores original units, including infinity for an
unrepresentable mean difference.
Large integer pairs subtract before floating-point conversion; reverse scoring
retains exact scale bounds, including fractional endpoints. Only the bounded
exceptional path uses Python integer arithmetic, and its means are rounded after
averaging and optional normalization.
Ordinary integer categories use native signed differences when safe. Selected
pair buffers are released before the next block, avoiding overlapping allocations.
Mahalanobis scoring uses the same row budget for missing-value detection,
complete-case means, centered covariance accumulation, and quadratic-form
evaluation. Complete-case selection happens inside each block, so enabling
`na_rm` does not copy the selected response matrix. Covariance decomposition
uses its real symmetric structure while retaining the inverse/pseudo-inverse
cutoff policy. Matrix workspaces depend on the row budget and square item count;
only the output and complete-case mask scale with the respondent count.
Guttman scoring likewise batches item means, difficulty-ordered selection,
valid-response counts, and error accumulation by respondent. Small categorical
scales use cumulative category counts within each batch, while high-cardinality
data use the same row bound with direct item-pair comparisons.
Split-half individual reliability generates the established seeded item splits
once, then reuses each bounded respondent block across them. The shared correlation
kernel centers owned selection buffers in place for both complete and missing
responses. Exceptional rows recover their original paired values from the input
block before rescaling, keeping source data unchanged. Missing-value detection
runs once per input block, and profiles with constant observed responses are
excluded before evaluating splits. Seeded split generation remains unchanged;
paired reductions retain each respondent's valid-split count.
Undefined Spearman–Brown corrections remain unavailable instead of
propagating infinity into screening or composites.
For exactly two observed pairs, including wider rows with missing responses,
the kernel obtains exact correlations from the relative ordering of each pair.
This avoids feeding rounded near-perfect negative correlations into the
Spearman–Brown correction.
Wider large-integer profiles shift by their observed minimum before conversion
to double precision. Unsigned differences preserve the full signed and unsigned
64-bit range; missing partners are excluded when choosing that minimum. Even–odd
scoring retains the original integer matrix until its bounded correlation work.
Complete-response onset detection derives stable sliding-window variability
from rolling means and bounded deviation buffers. Windows of at least 16 items
use cumulative first and second moments for integer-valued responses only when
all sums, squares, and products remain exact in double precision. The variance
numerator is formed before division, preserving constant-window variability
without cancellation. Other values retain the direct deviation calculation.
Large integer responses shift before floating-point conversion, and integer
blocks skip missing-value and infinity scans. Its changepoint test retains
only prefix and candidate-position workspaces instead of complete centered and
test-statistic matrices. Missing-response blocks compress rows into equal
retained-length groups and reuse the same bounded complete-response kernel.
Missing-value detection and eligibility checks share each input block, and the
rolling batch size accounts for its multiple workspaces. The changepoint series
is centered before accumulating moments to avoid cancellation from a common
baseline. Exceptionally large finite responses are scaled by powers of two;
logarithmic statistic comparisons retain the variance floor in the original
response units without overflowing or losing candidate ordering.
Person–total correlation calculates item-profile means and respondent
correlations in bounded batches as well. The shared kernel accepts the index's
undefined-correlation policy, so constant person or item profiles remain
unavailable rather than being assigned a synthetic score.
Strict person-total missing checks also use bounded batches and skip integer
inputs, which cannot contain missing values.
IRV, acquiescence, and response-style summaries share bounded row mean and
population-standard-deviation reductions. Both missing-value policies accumulate
in double precision and reuse the mean when centering. Complete blocks skip
masked reductions; blocks with mostly unavailable means center only the
remaining rows. Incomplete blocks retain observed counts, so an entirely
unavailable row returns an unavailable score without constructing a complete
boolean or centered workspace for the input. Exceptional finite rows use
power-of-two rescaling to recover overflowing means and overflowing or
underflowing variances. It also detects deviations small relative to their row
mean, where subtracting a rounded baseline can distort the remaining variation.
The repair first identifies constant observations, then shifts nearby values
by an observed value before calculating their centered deviations at a safe
scale. Wider ranges retain unshifted mean reductions to preserve cancellation
between positive and negative responses. Constant finite rows therefore have
exactly zero variability, while
nearly constant rows retain their representable differences. Infinite
observations retain unavailable deviations. The exceptional-moment detector and
the shifted scaling strategy are shared with row correlations, which preserve
each caller's zero-variance policy. Split IRV also
rescales overflowing section-score totals in bounded respondent blocks before
averaging them.
Integer row reductions skip missing masks. Ordinary integer totals accumulate
in native integer arithmetic when the complete total fits double precision
exactly. Exceptional 64-bit means sum high and low 32-bit parts separately,
combine only their reduced totals as Python integers, and round after division.
Their deviations use exact unsigned distances from each row's minimum before
conversion, preserving adjacent observations at large baselines without storing
a full object-valued response matrix.
Acquiescence normalizes observations before averaging when finite scale widths
overflow, are subnormal, or are very small relative to the baseline. Large integer
observations subtract an exact integer endpoint before conversion to float.
Balanced scoring selects both item polarities together and excludes incomplete
pairs from both halves; overflowing pair sums fall back to averaging the original
endpoints. Exceptionally narrow explicit bounds can make out-of-range response
proportions overflow; only those rows use decimal arithmetic to retain cancellation
before clipping the normalized mean.
Response-style endpoint comparisons resolve representable bounds once per call.
Integer midpoint intervals use exact rational arithmetic followed by integer
ceiling and floor, so fractional tolerances do not collapse adjacent categories.
Floating midpoint calculation adds before halving unless the sum overflows,
retaining constant subnormal scales. Explicit bounds are compared in double
precision even with single-precision response arrays. Integer proportions skip
missing scans, and the combined summary reuses these prepared comparisons.
Response-time summaries use the same reductions. Row medians own one bounded
buffer: incomplete rows are sorted with missing values last, while wide complete
rows use one partition and, for even lengths, the maximum of the lower half.
Strict reductions exclude incomplete rows before copying responses. Midpoint
arithmetic preserves large finite and subnormal results, promotes single-precision
endpoints, and averages large integer endpoints exactly before final rounding.
Median-based mixture preprocessing therefore does not duplicate the complete
timing matrix before fitting its respondent-level model, and log transformation
reuses the selected respondent medians.

For response checks against commit `d61be58`, a local 100,000-by-80 float64 matrix
with 10% missing responses and 40 selected items gave these medians (nine
alternating before/after runs after one warmup each, seed 20260927, Python 3.14.7,
NumPy 2.3.5, row-contiguous input):

| Operation | Before | After | Peak allocation before → after |
|-----------|-------:|------:|-------------------------------:|
| Attention checks, missing-as-pass proportions | 48.0 ms | 16.3 ms | 1.05 → 3.76 MiB |
| Missing rates, all items | 8.2 ms | 6.1 ms | 8.46 → 1.26 MiB |
| Missing rates, selected items with applicability | 33.0 ms | 21.4 ms | 40.66 → 3.56 MiB |

Reproduce the workload with `benchmarks/bench_response_checks.py --checks 40`.
Timing excludes allocation tracing. Attention checks trade additional bounded
workspace for throughput; missing-rate scoring reduces temporary allocation.
Results depend on layout and selection: column-contiguous all-item missing rates
were about 7% slower in the local comparison, while still using less memory.
Use `--order F` and `--checks 2` to explore these cases.

## Optional dependencies

All statistical functionality is available in the NumPy-only base install.
Chi-square quantiles use direct normal/exponential special cases for one or two
degrees of freedom, with the two-degree array path evaluated by NumPy. General
extreme lower tails are solved in logarithmic coordinates using the shared
gamma series, preserving probabilities and quantiles below the normal floating-point
range. These cases follow the [NIST gamma identities](https://dlmf.nist.gov/8.4);
regression tests include independently calculated high-precision lower-tail values.
Plotting remains optional and reports a centralized install hint from
`_optional_imports.py`: `pip install 'insufficient-effort[plot]'`.

## Parity and simulation

Performance scripts use `benchmarks/_measurement.py` for untraced timed repeats
and a separate peak-allocation call. Paired comparisons alternate operation order
and retain the last timed results for correctness checks. See
[Contributing](https://github.com/Cameron-Lyons/ier/blob/main/CONTRIBUTING.md#run-quality-checks)
for measurement details and historical-comparison guidance.

- Hand-locked regression fixtures live in `tests/test_golden_parity.py` and
  JSON under `tests/fixtures/parity/`.
- Detection-rate simulation: `benchmarks/bench_detection.py`.
- Throughput microbench: `benchmarks/bench_screen.py`.
- Multi-factor even–odd throughput and memory: `benchmarks/bench_evenodd.py`.
- Psychometric synonym missing-data throughput and memory: `benchmarks/bench_psychsyn.py`.
- Predefined semantic/MAD pair throughput and memory: `benchmarks/bench_pair_differences.py`.
- Mahalanobis covariance and distance throughput and memory: `benchmarks/bench_mahad.py`.
- Guttman error-scoring throughput and memory: `benchmarks/bench_guttman.py`.
- Split-half reliability throughput and memory: `benchmarks/bench_reliability.py`.
- Carelessness-onset throughput and memory: `benchmarks/bench_onset.py`.
- Person–total correlation throughput and memory: `benchmarks/bench_person_total.py`.
- Row-wise response reduction throughput and memory: `benchmarks/bench_row_reductions.py`.
- Missing-response and attention-check throughput and memory, including item
  subsets and applicability masks: `benchmarks/bench_response_checks.py`.
- Lz person-fit throughput and memory: `benchmarks/bench_lz.py`.
- Markov transition-entropy throughput and memory: `benchmarks/bench_markov.py`.
- Sequence indices and screening/composite workflows with configurable missingness:
  `benchmarks/bench_sequence_scoring.py`. Timing excludes allocation tracing;
  peak allocation is measured separately.
- Response-time mixture EM, scoring, and reusable cutoff sensitivity:
  `benchmarks/bench_response_time.py`.
- Screen/composite reduction memory: `benchmarks/bench_orchestration.py`.
- Reusable composite sensitivity analysis: `benchmarks/bench_composite.py`.
- Validated and atomic score and response-time archive loading and saving:
  `benchmarks/bench_archive.py`.
- Shared fixed and percentile flagging throughput and memory: `benchmarks/bench_flagging.py`.
- CLI JSON, CSV, and NPZ serialization: `benchmarks/bench_cli_output.py`.
  The same benchmark accepts `--compression gzip|bzip2|xz` for text formats.
- Delimited parsing with and without preamble rows: `benchmarks/bench_cli_input.py`.
