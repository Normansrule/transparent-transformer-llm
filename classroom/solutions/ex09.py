"""Exercise 09 - Sampling.   Lesson: stages/09_sampling/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def temperature_top_k(logits, temperature, k):
    """logits: (vocab,). Divide by temperature, keep only the k highest scores (the rest get probability 0),
    softmax what is left. Return the (vocab,) probabilities."""
    s = np.asarray(logits, dtype=float) / temperature
    s = np.where(s >= np.sort(s)[-k], s, -np.inf)
    e = np.exp(s - s.max())
    return e / e.sum()
