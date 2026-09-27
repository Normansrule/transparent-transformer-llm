"""
STAGE 5 - ATTENTION (THE CLOSE-UP)
==================================
Attention is how tokens talk to each other. For every token the layer asks:
"which EARLIER tokens matter for predicting what comes next, and how much?"

Each token vector is turned into three new vectors:

    Query (Q)  "what am I looking for?"
    Key   (K)  "what do I contain?"
    Value (V)  "what will I hand over if you pick me?"

    score[i, j] = Q_i . K_j / sqrt(d_head)     how well token j answers token i
    score[i, j] = -infinity   for j > i        CAUSAL MASK: no peeking at the future
    weights     = softmax(score)               each row now sums to 1
    out_i       = sum_j weights[i, j] * V_j    a weighted blend of the past

"Multi-head" just means we do this several times in parallel with smaller
vectors, so one head can track grammar while another tracks the city name.
"""
from __future__ import annotations

import numpy as np

from .layers import Layer, Linear


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)      # subtract max for numerical safety
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class CausalSelfAttention(Layer):
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator, out_std: float = 0.02):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model, rng)     # makes Q, K and V in one go
        self.proj = Linear(d_model, d_model, rng, std=out_std)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        H, hd = self.n_heads, self.d_head

        qkv = self.qkv.forward(x).reshape(B, T, 3, H, hd)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))   # each (B, H, T, hd)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)                  # (B, H, T, T)
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(future, -1e9, scores)                             # causal mask
        weights = softmax(scores, axis=-1)

        out = weights @ v                                                   # (B, H, T, hd)
        self.q, self.k, self.v, self.weights = q, k, v, weights             # remember for backward
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)                    # glue heads back together
        return self.proj.forward(out)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        B, T, D = dout.shape
        H, hd = self.n_heads, self.d_head
        q, k, v, w = self.q, self.k, self.v, self.weights

        dout = self.proj.backward(dout).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        dweights = dout @ v.transpose(0, 1, 3, 2)
        dv = w.transpose(0, 1, 3, 2) @ dout
        # softmax backward: each row's gradient, minus its weighted average
        dscores = w * (dweights - (dweights * w).sum(axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(hd)
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q

        dqkv = np.stack([dq, dk, dv], axis=2)                # (B, H, 3, T, hd)
        dqkv = dqkv.transpose(0, 3, 2, 1, 4).reshape(B, T, 3 * D)
        return self.qkv.backward(dqkv)
