# IER

A Python package for detecting Insufficient Effort Responding (IER) in survey data using various statistical indices and methods.

## Overview

When taking online surveys, participants sometimes respond to items without regard to their content. These types of responses, referred to as **insufficient effort responding** (IER) or **careless responding**, constitute significant problems for data quality, leading to distortions in data analysis and hypothesis testing.

The `ier` package provides solutions designed to detect such insufficient effort responses by allowing easy calculation of indices proposed in the literature. For a comprehensive review of these methods, see [Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

- **Multiple Detection Methods**: Supports 20+ indices for detecting careless responding
- **Flexible Input**: Works with lists, numpy arrays, pandas DataFrames, and polars DataFrames
- **Robust Implementation**: Handles missing data and edge cases
- **Type Hints**: Full type annotations for IDE support

## Installation

### From PyPI
```bash
pip install insufficient-effort
```

### From Source
```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
pip install -e .
```

### Optional Dependencies
For enhanced functionality (e.g., chi-squared outlier detection):
```bash
pip install insufficient-effort[full]
```

## Quick Start

```python
import numpy as np
from ier import irv, mahad, longstring, evenodd, psychsyn

# Sample survey data (rows = participants, columns = items)
data = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],  # Normal responding
    [3, 3, 3, 3, 3, 3, 3, 3],  # Straightlining
    [1, 5, 1, 5, 1, 5, 1, 5],  # Alternating pattern
])

# Intra-individual response variability (low = straightlining)
print("IRV:", irv(data))

# Mahalanobis distance (high = outlier)
print("Mahad:", mahad(data))

# Longest string of identical responses
print("Longstring:", longstring(data))
```

## Available Functions

### Consistency Indices

#### `evenodd(x, factors, diag=False)`
Computes even-odd consistency by correlating responses to even vs odd items within each factor.

```python
from ier import evenodd

data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
factors = [3, 3]  # Two factors with 3 items each
scores = evenodd(data, factors)
```

#### `psychsyn(x, critval=0.60, anto=False, diag=False)`
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

#### `mad(x, positive_items, negative_items, scale_max=None)`
Mean Absolute Difference between positively and negatively worded items. High MAD indicates careless responding (not attending to item direction).

```python
from ier import mad, mad_flag

# Columns 0,2 are positively worded; columns 1,3 are negatively worded
data = [
    [5, 1, 5, 1],  # Attentive: high on pos, low on neg
    [5, 5, 5, 5],  # Careless: ignores item direction
]
scores = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
scores, flags = mad_flag(data, positive_items=[0, 2], negative_items=[1, 3])
```

### Response Pattern Indices

#### `longstring(x, avg=False)`
Computes the longest (or average) run of identical consecutive responses.

```python
from ier import longstring

# Single string
longstring("AAABBBCCDAA")  # ('A', 3)

# Matrix of responses
data = [[1, 1, 1, 2, 3], [1, 2, 3, 4, 5]]
longstring(data)  # [('1', 3), ('1', 1)]
longstring(data, avg=True)  # [1.67, 1.0]
```

#### `irv(x, na_rm=True, split=False, num_split=1)`
Computes intra-individual response variability (standard deviation).

```python
from ier import irv

data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
scores = irv(data)  # High for varied, low for straightlining

# Split-half IRV
scores = irv(data, split=True, num_split=2)
```

#### `u3_poly(x, scale_min=None, scale_max=None)`
Proportion of extreme responses (at scale endpoints).

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

#### `response_pattern(x, scale_min=None, scale_max=None)`
Returns multiple response style indices at once.

```python
from ier import response_pattern

patterns = response_pattern(data, scale_min=1, scale_max=5)
# Returns dict with: extreme, midpoint, acquiescence, variability
```

### Statistical Outlier Detection

#### `mahad(x, flag=False, confidence=0.95, na_rm=False, method='chi2')`
Computes Mahalanobis distance for multivariate outlier detection.

```python
from ier import mahad, mahad_summary

data = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]]
distances = mahad(data)
distances, flags = mahad(data, flag=True, confidence=0.95)

# Methods: 'chi2', 'iqr', 'zscore'
distances, flags = mahad(data, flag=True, method='iqr')
```

#### `lz(x, difficulty=None, discrimination=None, theta=None, model='2pl')`
Standardized log-likelihood (lz) person-fit statistic based on Item Response Theory. Negative values indicate aberrant response patterns.

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
from ier import response_time, response_time_flag, response_time_consistency

times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]

avg_times = response_time(times, metric='mean')
med_times = response_time(times, metric='median')
min_times = response_time(times, metric='min')

# Flag fast responders
flags = response_time_flag(times, threshold=1.0)

# Coefficient of variation (low = suspiciously uniform)
cv = response_time_consistency(times)
```

### Composite Index

#### `composite(x, indices=None, method='mean', standardize=True)`
Combines multiple IER indices into a single composite score. Higher scores indicate greater likelihood of careless responding.

```python
from ier import composite, composite_flag, composite_summary

data = [
    [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],  # Normal
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # Straightliner
]

# Default: combines IRV, longstring, Mahalanobis, psychsyn, person-total
scores = composite(data)

# Select specific indices
scores = composite(data, indices=['irv', 'longstring', 'mahad'])

# Different combination methods
scores = composite(data, method='sum')   # Sum of z-scores
scores = composite(data, method='max')   # Maximum z-score

# Flag careless responders
scores, flags = composite_flag(data, threshold=1.5)
scores, flags = composite_flag(data, percentile=95.0)

# Detailed summary with individual index scores
summary = composite_summary(data)
print(summary['indices_used'])  # ['irv', 'longstring', 'mahad', ...]
print(summary['indices'])       # Dict of individual index scores
```

## Working with DataFrames

The package works with pandas and polars DataFrames:

```python
import pandas as pd
import polars as pl
from ier import irv

# Pandas
df_pandas = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
scores = irv(df_pandas)

# Polars
df_polars = pl.DataFrame([[1, 2, 3], [4, 5, 6]])
scores = irv(df_polars)
```

## Handling Missing Data

Most functions handle NaN values appropriately:

```python
import numpy as np
from ier import irv, mahad

data = np.array([
    [1, 2, np.nan, 4],
    [np.nan, 2, 3, 4],
    [1, 2, 3, 4]
])

irv_scores = irv(data, na_rm=True)
mahad_scores = mahad(data, na_rm=True)
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
