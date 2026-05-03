# IER

A Python package for detecting Insufficient Effort Responding (IER) in survey data using various statistical indices and methods.

## Overview

When taking online surveys, participants sometimes respond to items without regard to their content. These types of responses, referred to as **insufficient effort responding** (IER) or **careless responding**, constitute significant problems for data quality, leading to distortions in data analysis and hypothesis testing.

The `ier` package provides solutions designed to detect such insufficient effort responses by allowing easy calculation of indices proposed in the literature. For a comprehensive review of these methods, see [Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

- **Multiple Detection Methods**: Consistency, response-pattern, response-style, outlier, response-time, and attention-check indices
- **Workflow APIs**: `screen()` runs multiple indices at once; `composite()` combines selected signals into a single score
- **Flexible Input**: Works with lists, tuples, numpy arrays, and array-compatible DataFrame objects
- **Robust Implementation**: Handles missing data and records per-index screening errors where possible
- **Type Hints**: Full type annotations and typed result dictionaries for IDE support

## Installation

### From PyPI
```bash
pip install insufficient-effort
```

### From Source
```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
python -m pip install -e .
```

### Optional Dependencies
Base installation includes NumPy-only indices. Install extras for SciPy-backed
statistics and plotting:

```bash
python -m pip install "insufficient-effort[full]"
python -m pip install "insufficient-effort[plot]"
python -m pip install "insufficient-effort[full,plot]"
```

The `full` extra is required for `mahad(method="chi2")`,
`mahad_qqplot()`, and `response_time_mixture()`. The default `screen()` and
`composite()` index sets include `mahad`, so install `full` or pass an
explicit NumPy-only `indices=[...]` list.

## Quick Start

```python
import numpy as np
from ier import composite, irv, screen

# Sample survey data (rows = participants, columns = items)
data = np.array([
    [1, 2, 3, 4, 5, 4],
    [2, 3, 4, 3, 2, 1],
    [3, 3, 3, 3, 3, 3],  # Straightlining
    [1, 5, 1, 5, 1, 5],  # Alternating pattern
    [5, 4, 3, 2, 1, 2],
    [2, 2, 3, 4, 4, 5],
    [4, 5, 4, 3, 2, 1],
    [1, 2, 1, 2, 3, 4],
], dtype=float)

# Intra-individual response variability (low = straightlining)
print("IRV:", irv(data))

# Run a NumPy-only screening set
indices = [
    "irv",
    "longstring",
    "longstring_pattern",
    "markov",
    "person_total",
    "u3_poly",
    "midpoint",
    "acquiescence",
]
result = screen(data, indices=indices, scale_min=1, scale_max=5)
print("Indices used:", result["indices_used"])
print("Flag counts:", result["flag_counts"])

# Combine selected indices into one score
scores = composite(data, indices=["irv", "longstring", "person_total", "markov"])
print("Composite:", scores)
```

## Available Functions

### Workflow APIs

#### `screen(x, indices=None, percentile=95.0, ...)`
Runs multiple IER indices, flags respondents by percentile thresholds, and
returns structured scores, flags, flag counts, per-index errors, and summary
statistics.

```python
from ier import screen

data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [1, 5, 1, 5, 1]]
result = screen(
    data,
    indices=["irv", "longstring", "longstring_pattern", "u3_poly", "midpoint"],
    scale_min=1,
    scale_max=5,
)

print(result["scores"])       # Dict of index name -> score array
print(result["flags"])        # Dict of index name -> boolean flag array
print(result["flag_counts"])  # Total flags per respondent
print(result["errors"])       # Indices skipped because of missing config or invalid data
```

Default screen indices are `irv`, `longstring`, `longstring_pattern`, `mahad`,
`psychsyn`, `person_total`, `markov`, `u3_poly`, `midpoint`, and
`acquiescence`. Optional requested indices include `evenodd`, `mad`, and `lz`;
`evenodd` requires `evenodd_factors`, and `mad` requires positive and negative
item lists.

#### Plotting screen results
The plotting helpers require the `plot` extra.

```python
from ier import plot_distributions, plot_flag_counts, plot_flagged_heatmap

fig = plot_distributions(result)
heatmap = plot_flagged_heatmap(result)
counts = plot_flag_counts(result)
```

### Consistency Indices

#### `evenodd(x, factors, diag=False)`
Computes even-odd consistency by correlating responses to even vs odd items within each factor.

```python
from ier import evenodd

data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
factors = [3, 3]  # Two factors with 3 items each
scores = evenodd(data, factors)
```

#### `psychsyn(x, critval=0.60, anto=False, diag=False, resample_na=False, random_seed=None)` / `psychant(...)`
Identifies highly correlated item pairs and computes within-person correlations.

```python
from ier import psychsyn, psychant

data = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
scores = psychsyn(data, critval=0.5)  # Synonyms
scores = psychant(data, critval=-0.5)  # Antonyms
```

#### `individual_reliability(x, n_splits=100, random_seed=None)`
Estimates response consistency using repeated split-half correlations.

```python
from ier import individual_reliability, individual_reliability_flag

data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 3, 3]]
reliability = individual_reliability(data, n_splits=50)
flags = individual_reliability_flag(data, threshold=0.3)
```

#### `person_total(x, na_rm=True)`
Correlates each person's responses with the sample mean response pattern.

```python
from ier import person_total

data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
scores = person_total(data)  # [1.0, -1.0, 1.0]
```

#### `semantic_syn(x, item_pairs, anto=False)` / `semantic_ant(x, item_pairs)`
Computes consistency for predefined semantic synonym/antonym pairs.

```python
from ier import semantic_syn, semantic_ant

data = [[1, 1, 5, 5], [1, 2, 5, 4]]
pairs = [(0, 1), (2, 3)]  # Predefined synonym pairs
scores = semantic_syn(data, pairs)
```

#### `guttman(x, na_rm=True, normalize=True)`
Counts response reversals relative to item difficulty ordering.

```python
from ier import guttman, guttman_flag

data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
errors = guttman(data)
flags = guttman_flag(data, threshold=0.5)
```

#### `mad(x, positive_items=None, negative_items=None, item_pairs=None, scale_max=None, na_rm=True)`
Mean Absolute Difference between positively and negatively worded items after
reverse-scoring the negative items. High MAD indicates careless responding
(not attending to item direction).

```python
from ier import mad, mad_flag

# Columns 0,2 are positively worded; columns 1,3 are negatively worded
data = [
    [5, 1, 5, 1],  # Attentive: high on pos, low on neg
    [5, 5, 5, 5],  # Careless: ignores item direction
]
scores = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
scores, flags = mad_flag(data, positive_items=[0, 2], negative_items=[1, 3])

# Equivalent paired-item form
scores = mad(data, item_pairs=[(0, 1), (2, 3)], scale_max=5)
```

### Response Pattern and Style Indices

#### `longstring(messages, avg=False)`
Computes the longest or average run of identical consecutive characters in a
single string, list of strings, or one-dimensional numpy array. Numeric matrix
longstring scoring is available through `screen(..., indices=["longstring"])`
and `composite(..., indices=["longstring"])`.

```python
from ier import longstring

longstring("AAABBBCCDAA")  # ('A', 3)
longstring("AAABBBCCDAA", avg=True)  # 2.2
longstring(["11123", "12345", "33333"])  # [('1', 3), ('1', 1), ('3', 5)]
```

#### `longstring_pattern(x, max_pattern_length=5, na_rm=True)`
Detects repeating numeric sub-patterns such as 1-2-1-2 or 1-2-3-1-2-3.

```python
from ier import longstring_pattern

data = [[1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6]]
pattern_lengths = longstring_pattern(data)
```

#### `irv(x, na_rm=True, split=False, num_split=1, split_points=None)`
Computes intra-individual response variability (standard deviation). Low values
often indicate straightlining; high values may indicate random responding.

```python
from ier import irv

data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
scores = irv(data)
split_scores = irv(data, split=True, num_split=2)
```

#### `markov(x, na_rm=True)`
Computes first-order transition entropy for each response sequence. Low entropy
indicates highly predictable patterns.

```python
from ier import markov, markov_flag, markov_summary

data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
entropy = markov(data)
scores, flags = markov_flag(data, percentile=5.0)
summary = markov_summary(data)
```

#### `u3_poly(x, scale_min=None, scale_max=None)`
Proportion of responses at the scale endpoints.

```python
from ier import u3_poly

data = [[1, 5, 1, 5, 3], [3, 3, 3, 3, 3]]
extreme = u3_poly(data, scale_min=1, scale_max=5)
```

#### `midpoint_responding(x, scale_min=None, scale_max=None, tolerance=0.0)`
Proportion of midpoint responses.

```python
from ier import midpoint_responding

data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
mid = midpoint_responding(data, scale_min=1, scale_max=5)  # [0.2, 1.0]
```

#### `acquiescence(x, scale_min=None, scale_max=None, positive_items=None, negative_items=None, na_rm=True)`
Computes normalized agreement tendency. With balanced positive/negative item
lists, it reverse-scores negative items to isolate agreement bias.

```python
from ier import acquiescence, acquiescence_flag

data = [[5, 5, 5, 5], [3, 3, 3, 3], [1, 1, 1, 1]]
scores = acquiescence(data, scale_min=1, scale_max=5)
scores, flags = acquiescence_flag(data, scale_min=1, scale_max=5, threshold=0.9)
```

#### `response_pattern(x, scale_min=None, scale_max=None)`
Returns multiple response style indices at once.

```python
from ier import response_pattern

data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
patterns = response_pattern(data, scale_min=1, scale_max=5)
# Returns dict with: extreme, midpoint, acquiescence, variability
```

#### `infrequency(x, item_indices, expected_responses, proportion=False)`
Counts failed attention-check or bogus items.

```python
from ier import infrequency, infrequency_flag

data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
failures = infrequency(data, item_indices=[0, 2], expected_responses=[5, 1])
scores, flags = infrequency_flag(data, [0, 2], [5, 1], threshold=2)
```

#### `onset(x, window_size=10, min_items=20, na_rm=True)`
Detects the item index where a respondent's pattern shifts toward careless
responding using running IRV and changepoint detection.

```python
from ier import onset, onset_flag

long_survey_data = [
    [1, 2, 3, 4, 5] * 4,
    [2, 4, 3, 5, 1] * 2 + [3] * 10,
]
onset_indices = onset(long_survey_data, window_size=10, min_items=20)
flags = onset_flag(long_survey_data, window_size=10, min_items=20)
```

### Statistical Outlier Detection

#### `mahad(x, flag=False, confidence=0.95, na_rm=False, method='chi2')`
Computes Mahalanobis distance for multivariate outlier detection. The default
`method="chi2"` requires the `full` extra; `method="iqr"` and `method="zscore"`
can be used without SciPy for flagging.

```python
from ier import mahad, mahad_qqplot

data = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]]
distances = mahad(data, method='iqr')
distances, flags = mahad(data, flag=True, method='iqr')

# Requires insufficient-effort[full]
distances, flags = mahad(data, flag=True, confidence=0.95, method='chi2')
theoretical, observed = mahad_qqplot(data)
```

#### `lz(x, difficulty=None, discrimination=None, theta=None, model='2pl')`
Standardized log-likelihood (lz) person-fit statistic based on Item Response Theory. Negative values indicate aberrant response patterns.

When `difficulty`, `discrimination`, or `theta` are omitted, `ier` estimates
them from the same response matrix as a convenience fallback. Treat the
resulting lz values as screening statistics rather than fully calibrated IRT
person-fit tests. Polytomous responses are dichotomized at the observed
midpoint before scoring, which discards category information.

```python
from ier import lz, lz_flag

# Binary response data (0/1)
data = [
    [1, 1, 1, 0, 0, 0],  # Normal pattern
    [0, 0, 0, 1, 1, 1],  # Aberrant pattern (fails easy, passes hard)
]
scores = lz(data)  # Negative = suspicious
scores, flags = lz_flag(data, threshold=-1.96)

# Use 1PL (Rasch) model
scores = lz(data, model='1pl')

# Provide custom item parameters
scores = lz(data, difficulty=[-1, -0.5, 0, 0.5, 1, 1.5])
```

### Response Time Indices

#### `response_time(times, metric='median')`
Computes response time statistics per person.

```python
from ier import (
    response_time,
    response_time_consistency,
    response_time_flag,
    response_time_mixture,
)

times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]

avg_times = response_time(times, metric='mean')
med_times = response_time(times, metric='median')
min_times = response_time(times, metric='min')

# Flag fast responders
flags = response_time_flag(times, threshold=1.0)

# Coefficient of variation (low = suspiciously uniform)
cv = response_time_consistency(times)

# Requires insufficient-effort[full]
fast_component_probability = response_time_mixture(times, random_seed=42)
```

### Composite Index

#### `composite(x, indices=None, method='mean', standardize=True, return_diagnostics=False, ...)`
Combines multiple IER indices into a single composite score. Higher scores
indicate greater likelihood of careless responding.

```python
from ier import composite, composite_flag, composite_probability, composite_summary

data = [
    [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],  # Normal
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # Straightliner
    [1, 5, 1, 5, 1, 5, 1, 5, 1, 5],  # Alternating
]
base_indices = ['irv', 'longstring', 'person_total', 'markov']

# Select NumPy-only indices
scores = composite(data, indices=base_indices)

# Include configured indices
scores = composite(
    data,
    indices=['irv', 'longstring', 'mad'],
    mad_positive_items=[0, 2, 4],
    mad_negative_items=[1, 3, 5],
    mad_scale_max=5,
)

# Different combination methods
scores = composite(data, indices=base_indices, method='sum')  # Sum of z-scores
scores = composite(data, indices=base_indices, method='max')  # Maximum z-score
scores = composite(data, method='best_subset')  # IRV, longstring, lz; MAD if configured

# Flag careless responders
scores, flags = composite_flag(data, indices=base_indices, threshold=1.5)
scores, flags = composite_flag(data, indices=base_indices, percentile=95.0)

# Optional diagnostics for skipped indices
scores, diagnostics = composite(
    data,
    indices=['irv', 'mad'],
    return_diagnostics=True,
)

# Detailed summary with individual index scores
summary = composite_summary(data, indices=base_indices)
print(summary['indices_used'])  # ['irv', 'longstring', 'person_total', 'markov']
print(summary['indices'])       # Dict of individual index scores

# Sample-relative logistic transform, not a calibrated probability
probability_scores = composite_probability(data, indices=['irv', 'longstring'])
```

`composite_probability()` returns a logistic transform of the sample-relative
standardized composite score. It is bounded between 0 and 1, but it is not a
calibrated probability of IER unless you validate and calibrate it against
labeled data from a comparable survey context.

Default composite indices are `irv`, `longstring`, `mahad`, `psychsyn`, and
`person_total`. Additional allowed composite indices are `evenodd`, `mad`,
`markov`, `longstring_pattern`, and `lz`; `u3_poly`, `midpoint`, and
`acquiescence` are screening-only response-style indices.

## Working with DataFrames

Functions accept array-like inputs. DataFrame objects that expose `__array__`
can be passed directly; otherwise pass their numpy representation.

```python
import pandas as pd
import polars as pl
from ier import irv

# Pandas
df_pandas = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
scores = irv(df_pandas)

# Polars
df_polars = pl.DataFrame([[1, 2, 3], [4, 5, 6]])
scores = irv(df_polars.to_numpy())
```

## Handling Missing Data

Most functions handle NaN values appropriately:

```python
import numpy as np
from ier import irv, mahad

data = np.array([
    [1, 2],
    [2, 3],
    [np.nan, 4],
    [3, 4],
    [10, 10],
], dtype=float)

irv_scores = irv(data, na_rm=True)
mahad_scores = mahad(data, na_rm=True, method="iqr")
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
quality checks, and release instructions.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{ier2026,
  title={IER: Python package for detecting Insufficient Effort Responding},
  author={Lyons, Cameron},
  year={2026},
  url={https://github.com/Cameron-Lyons/ier}
}
```

## References

- Curran, P. G. (2016). Methods for the detection of carelessly invalid responses in survey data. *Journal of Experimental Social Psychology*, 66, 4-19.
- Dunn, A. M., Heggestad, E. D., Shanock, L. R., & Theilgard, N. (2018). Intra-individual response variability as an indicator of insufficient effort responding. *Journal of Business and Psychology*, 33(1), 105-121.
