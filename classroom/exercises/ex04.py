"""Exercise 04 - Transformer.   Lesson: stages/04_transformer/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def layer_norm(x, gain, bias, eps=1e-5):
    """x: (T, d). For EACH ROW: subtract its mean, divide by sqrt(variance + eps), then multiply by gain and add bias."""
    raise NotImplementedError("your code here")
