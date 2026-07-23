# IER

Python library for detecting **Insufficient Effort Responding (IER)** / careless
responding in survey data.

## Install

```bash
pip install insufficient-effort
```

Optional extras:

```bash
pip install "insufficient-effort[full]"   # SciPy-backed mahad chi2 flagging, RT mixture
pip install "insufficient-effort[plot]"   # matplotlib visualizations
```

Base install is NumPy-only. Default `screen()` and `composite()` work without SciPy.

## Quick start

```python
import numpy as np
from ier import composite, screen

data = np.array([
    [1, 2, 3, 4, 5, 4],
    [3, 3, 3, 3, 3, 3],  # straightlining
    [1, 5, 1, 5, 1, 5],
], dtype=float)

result = screen(data, scale_min=1, scale_max=5)
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
