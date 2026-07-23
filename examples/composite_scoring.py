"""Composite scoring example with explicit NumPy-only indices."""

import numpy as np

from ier import composite, composite_probability

rng = np.random.default_rng(11)
data = rng.integers(1, 6, size=(50, 10)).astype(float)
data[2, :] = 3.0

indices = ["irv", "longstring", "person_total", "markov", "guttman"]
scores = composite(data, indices=indices)
probs = composite_probability(data, indices=indices)

print("composite head:", np.round(scores[:5], 3))
print("uncalibrated logistic head:", np.round(probs[:5], 3))
print(
    "note: logistic values rank respondents within-sample; "
    "they are not calibrated IER probabilities."
)
