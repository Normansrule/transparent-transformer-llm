"""Exercise 07 - Backpropagation.   Lesson: stages/07_backpropagation/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def linear_backward(x, W, dy):
    """Forward pass was  y = x @ W   with x: (T, d_in), W: (d_in, d_out).
    Given dy = dLoss/dy of shape (T, d_out), return (dx, dW): the gradients for the input and for the weights."""
    return dy @ W.T, x.T @ dy
