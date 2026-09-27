"""Exercise 01 - Input.   Lesson: stages/01_input/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def to_bytes(text: str) -> list[int]:
    """Return the UTF-8 bytes of `text` as a list of integers.   to_bytes("Hi") -> [72, 105]"""
    return list(text.encode("utf-8"))
