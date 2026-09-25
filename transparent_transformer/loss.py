"""
STAGE 6/7 - THE LOSS: ONE NUMBER THAT SAYS "HOW WRONG WAS THAT?"
================================================================
The model outputs a score (logit) for every token in the vocabulary.
softmax turns scores into probabilities. The loss is simply

    loss = -log( probability the model gave to the CORRECT next token )

  * gave it 100%   -> loss 0.0    perfect
  * gave it  50%   -> loss 0.69
  * gave it   1%   -> loss 4.6    very surprised
  * uniform guess over 768 tokens -> loss ln(768) = 6.64   (where training starts)

The gradient of this loss with respect to the logits is beautifully simple:

    dlogits = probabilities - one_hot(correct token)

"Push the correct token's score up, push everyone else's down, in proportion
to how much probability you wasted on them." That vector is where
backpropagation (stage 7) begins.
"""
from __future__ import annotations

import numpy as np

from .attention import softmax


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None):
    """Returns (loss, dlogits). `mask` (B, T) is 1 where a position counts, 0 where it is ignored."""
    B, T, V = logits.shape
    if mask is None:
        mask = np.ones((B, T), dtype=logits.dtype)
    n = max(mask.sum(), 1.0)
    logp = log_softmax(logits)
    picked = np.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]   # log p(correct token)
    loss = -(picked * mask).sum() / n

    dlogits = softmax(logits)                                   # probabilities ...
    np.put_along_axis(dlogits, targets[..., None],
                      np.take_along_axis(dlogits, targets[..., None], axis=-1) - 1.0, axis=-1)  # ... minus one-hot
    dlogits *= (mask / n)[..., None]
    return float(loss), dlogits.astype(logits.dtype)


def sequence_logprob(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum of log p(token) over the masked positions of each sequence. Shape (B,). Used by DPO."""
    picked = np.take_along_axis(log_softmax(logits), targets[..., None], axis=-1)[..., 0]
    return (picked * mask).sum(axis=-1)
