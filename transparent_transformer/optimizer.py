"""
STAGE 7 (part) - THE OPTIMIZER: TURNING GRADIENTS INTO BETTER WEIGHTS
=====================================================================
Backpropagation tells us, for every weight, "if you nudge me up, the loss
goes up by this much". The optimizer then nudges every weight the OTHER way.

Plain gradient descent:      w = w - lr * grad

AdamW (what almost every LLM uses) adds two ideas:
  * momentum  (m): average recent gradients so noise cancels out
  * scaling   (v): take bigger steps for weights whose gradients are tiny,
                   smaller steps for weights whose gradients are huge
  * weight decay : gently pull weights toward zero to discourage memorising
"""
from __future__ import annotations

import math

import numpy as np


class AdamW:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-3, betas=(0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params, self.lr, self.betas, self.eps, self.wd = params, lr, betas, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray], lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        b1, b2 = self.betas
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            if p.ndim > 1:                                # decay matrices, not biases or gains
                p *= 1 - lr * self.wd
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)   # in place: the model sees the update


def clip_gradients(grads: dict[str, np.ndarray], max_norm: float = 1.0) -> float:
    """If the overall gradient is too long, shrink it. Prevents one bad batch from wrecking the model."""
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
    if total > max_norm:
        for g in grads.values():
            g *= max_norm / (total + 1e-6)
    return total


def cosine_schedule(step: int, total: int, peak: float, warmup: int = 50, floor: float = 0.1) -> float:
    """Learning rate: ramp up quickly, then glide down along a cosine."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
