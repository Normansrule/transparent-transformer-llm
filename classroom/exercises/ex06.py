"""Exercise 06 - Pretraining.   Lesson: stages/06_pretraining/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def next_token_loss(probs, targets):
    """probs: (T, vocab), each row a probability distribution. targets: (T,) the correct next-token ids.
    Return the average of  -log(probability given to the correct token)."""
    raise NotImplementedError("your code here")
