"""Exercise 02 - Tokenization.   Lesson: stages/02_tokenizer/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`. The heart of Byte Pair Encoding (BPE).
    merge([1, 2, 3, 1, 2], (1, 2), 9) -> [9, 3, 9]"""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == tuple(pair):
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
