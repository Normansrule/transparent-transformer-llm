"""Exercise 05 - Attention.   Lesson: stages/05_attention_closeup/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def causal_attention_weights(scores):
    """scores: (T, T), row i = how much token i likes each token j.
    1. hide the future: every entry with j > i must end up with weight exactly 0
    2. softmax each row so it sums to 1.   Return the (T, T) weights."""
    raise NotImplementedError("your code here")
