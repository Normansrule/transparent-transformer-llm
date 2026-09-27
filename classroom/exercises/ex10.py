"""Exercise 10 - Output.   Lesson: stages/10_output/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def generate(next_token, ids, end_id, max_new=20):
    """The autoregressive loop. `next_token(ids)` returns the next token id for a list of ids.
    Keep calling it and appending, stop when it returns `end_id` (do not include end_id) or after max_new tokens.
    Return ONLY the new ids."""
    raise NotImplementedError("your code here")
