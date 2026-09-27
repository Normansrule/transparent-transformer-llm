"""
STAGE 5 (part) - THE BASIC BUILDING BLOCKS
==========================================
Every layer here has exactly two methods:

    forward(x)      data flows DOWN the network, the layer remembers what it saw
    backward(dout)  blame flows UP the network; given "how much did the loss
                    change when my OUTPUT changed" (dout) it computes
                      1. how much the loss changes when its WEIGHTS change (self.grads)
                      2. how much the loss changes when its INPUT changes  (returned)

There is no automatic differentiation in this repository. Every gradient is
written out by hand so you can read it. tests/test_gradients.py proves each
one is correct by comparing against a brute-force numerical estimate.
"""
from __future__ import annotations

import numpy as np


class Layer:
    """Tiny base class: a bag of named weights and their gradients."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}


class Linear(Layer):
    """y = x @ W + b      the workhorse: a learned weighted sum."""

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, std: float = 0.02):
        super().__init__()
        self.params["W"] = (rng.standard_normal((d_in, d_out)) * std).astype(np.float32)
        self.params["b"] = np.zeros(d_out, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x2 = self.x.reshape(-1, self.x.shape[-1])          # (B*T, d_in)
        d2 = dout.reshape(-1, dout.shape[-1])              # (B*T, d_out)
        self.grads["W"] = x2.T @ d2                        # dL/dW
        self.grads["b"] = d2.sum(axis=0)                   # dL/db
        return dout @ self.params["W"].T                   # dL/dx


class LayerNorm(Layer):
    """Re-centre and re-scale each token vector so numbers stay well-behaved."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.params["g"] = np.ones(d, dtype=np.float32)    # learned gain
        self.params["b"] = np.zeros(d, dtype=np.float32)   # learned bias
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.std = np.sqrt(var + self.eps)
        self.xhat = (x - mu) / self.std
        return self.xhat * self.params["g"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n = dout.shape[-1]
        self.grads["g"] = (dout * self.xhat).reshape(-1, n).sum(axis=0)
        self.grads["b"] = dout.reshape(-1, n).sum(axis=0)
        dxhat = dout * self.params["g"]
        return (dxhat
                - dxhat.mean(axis=-1, keepdims=True)
                - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)) / self.std


class GELU(Layer):
    """Gaussian Error Linear Unit (GELU): a smooth on/off switch for each number.

    Without a non-linearity like this, stacking layers would collapse into one
    big matrix multiply and the network could only learn straight lines.
    """
    A = np.sqrt(2.0 / np.pi)
    B = 0.044715

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.t = np.tanh(self.A * (x + self.B * x ** 3))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, t = self.x, self.t
        dy = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * self.A * (1.0 + 3.0 * self.B * x ** 2)
        return dout * dy


class MLP(Layer):
    """Multi-Layer Perceptron (MLP): expand -> GELU -> shrink.

    Attention moves information BETWEEN tokens.
    The MLP then thinks about each token ON ITS OWN. Much of what a model
    "knows" (Los Angeles -> sunny) is stored in these weights.
    """

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator, out_std: float = 0.02):
        super().__init__()
        self.up = Linear(d_model, d_ff, rng)
        self.act = GELU()
        self.down = Linear(d_ff, d_model, rng, std=out_std)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.down.forward(self.act.forward(self.up.forward(x)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.up.backward(self.act.backward(self.down.backward(dout)))
