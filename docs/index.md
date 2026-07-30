# IER

Python library for detecting **Insufficient Effort Responding (IER)** / careless
responding in survey data.

## Install

```bash
pip install insufficient-effort
```

Optional extras:

```bash
pip install "insufficient-effort[plot]"
```

The NumPy-only base install includes chi-square flagging, Q-Q quantiles, IRT
theta estimation, and response-time mixture scoring. The legacy `full` extra
is retained as an empty compatibility alias.

## Quick start

```python
import numpy as np
from ier import IndexOptions, composite, screen

data = np.array([
    [1, 2, 3, 4, 5, 4],
    [3, 3, 3, 3, 3, 3],
    [1, 5, 1, 5, 1, 5],
], dtype=float)

result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
print(result["flag_counts"])

scores = composite(data, indices=["irv", "longstring", "person_total"])
print(scores)
```

## Learn more

- [Getting started](getting-started.md)
- [Index catalog](indices.md)
- [Screening workflow](workflows/screening.md)
- [Composite guidance](workflows/composite.md)
- [Threshold guidance](thresholds.md)
- [R package notes](r-comparison.md)
- [API reference](api.md)
- [Changelog](changelog.md)

## Citation

```bibtex
@software{ier2026,
  title={IER: Python package for detecting Insufficient Effort Responding},
  author={Lyons, Cameron},
  year={2026},
  url={https://github.com/Cameron-Lyons/ier}
}
```
