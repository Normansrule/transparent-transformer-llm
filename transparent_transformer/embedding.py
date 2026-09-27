"""
STAGE 3 - EMBEDDING
===================
A token id like 291 is just a name tag. It carries no meaning: token 291 is
not "bigger" than token 290. The embedding layer swaps each id for a vector
of `d_model` numbers that the network is free to shape during training.

    token ids  (T,)            [302, 275, 261, 291]
        |  look up one row per id in a (vocab_size x d_model) table
        v
    token vectors (T, d_model)
        +  position vectors (T, d_model)   <- "I am the 1st / 2nd / 3rd token"
        =
    x  (T, d_model)   the "residual stream" that flows through the transformer

Why add positions? Attention (stage 5) looks at all tokens at once and has no
built-in sense of order. Without positions, "dog bites man" and "man bites
dog" would look identical.
"""
from __future__ import annotations

import numpy as np

from .layers import Layer


class Embedding(Layer):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, rng: np.random.Generator):
        super().__init__()
        self.params["tok"] = (rng.standard_normal((vocab_size, d_model)) * 0.02).astype(np.float32)
        self.params["pos"] = (rng.standard_normal((context_length, d_model)) * 0.02).astype(np.float32)

    def forward(self, ids: np.ndarray) -> np.ndarray:
        """ids: (B, T) integers  ->  (B, T, d_model) floats"""
        self.ids = ids
        T = ids.shape[1]
        self.tok_vectors = self.params["tok"][ids]       # a table lookup, nothing more
        self.pos_vectors = self.params["pos"][:T]
        return self.tok_vectors + self.pos_vectors

    def backward(self, dout: np.ndarray) -> None:
        """Only the rows that were actually looked up receive any gradient."""
        T = self.ids.shape[1]
        dtok = np.zeros_like(self.params["tok"])
        np.add.at(dtok, self.ids.reshape(-1), dout.reshape(-1, dout.shape[-1]))
        dpos = np.zeros_like(self.params["pos"])
        dpos[:T] = dout.sum(axis=0)
        self.grads["tok"] = dtok
        self.grads["pos"] = dpos
