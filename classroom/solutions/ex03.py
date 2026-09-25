"""Exercise 03 - Embedding.   Lesson: stages/03_embedding/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def embed(ids, token_table, position_table):
    """ids: (T,) integers. token_table: (vocab, d). position_table: (context, d).
    Return the (T, d) array: each token's row PLUS the row for its position (0, 1, 2, ...)."""
    ids = np.asarray(ids)
    return token_table[ids] + position_table[: len(ids)]
