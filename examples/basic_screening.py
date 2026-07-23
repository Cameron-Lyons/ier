"""Basic multi-index screening example."""

import numpy as np

from ier import IndexOptions, screen

rng = np.random.default_rng(7)
data = rng.integers(1, 6, size=(40, 12)).astype(float)
data[0, :] = 3.0  # straightliner
data[1, :] = np.tile([1.0, 5.0], 6)  # alternating

result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
print("indices:", result["indices_used"])
print("top flagged respondents:", np.argsort(result["flag_counts"])[::-1][:5])
print("flag counts:", result["flag_counts"][:5], "...")
