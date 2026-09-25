"""
STAGE 9 - SAMPLING: FROM SCORES TO ONE CHOSEN TOKEN
===================================================
The transformer never outputs a word. It outputs `vocab_size` numbers
(logits), one per token, saying how plausible each would be as the next token.
Sampling is the small, separate step that turns those scores into a choice.

    logits --(/ temperature)--> --(top-k)--> --(top-p)--> softmax --> roll the dice --> token id

  temperature  < 1 sharpens the distribution (safer, more repetitive)
               > 1 flattens it             (more surprising, more mistakes)
               = 0 means "always take the top one" (greedy)
  top-k        keep only the k best-scoring tokens
  top-p        keep the smallest set of tokens whose probabilities add up to p

Then the chosen token is appended to the input and THE WHOLE MODEL RUNS AGAIN
to get the next one. That loop is called autoregressive generation. It is the
reason LLM text appears one piece at a time.
"""
from __future__ import annotations

import numpy as np

from .attention import softmax


def sample_next(logits: np.ndarray, temperature: float = 0.8, top_k: int | None = 20,
                top_p: float | None = 0.95, rng: np.random.Generator | None = None) -> tuple[int, dict]:
    """logits: (vocab_size,). Returns (token id, details-for-visualisation)."""
    logits = logits.astype(np.float64)
    raw_probs = softmax(logits)
    info = {"raw_probs": raw_probs}

    if temperature <= 0:                                   # greedy decoding
        tok = int(np.argmax(logits))
        info["final_probs"] = np.eye(len(logits))[tok]
        return tok, info

    scaled = logits / temperature
    keep = np.ones_like(scaled, dtype=bool)
    if top_k is not None and top_k < len(scaled):
        kth_best = np.sort(scaled)[-top_k]
        keep &= scaled >= kth_best
    if top_p is not None and top_p < 1.0:
        p = softmax(np.where(keep, scaled, -np.inf))
        order = np.argsort(-p)
        cumulative = np.cumsum(p[order])
        n_keep = int(np.searchsorted(cumulative, top_p) + 1)   # smallest set reaching top_p
        nucleus = np.zeros_like(keep)
        nucleus[order[:n_keep]] = True
        keep &= nucleus

    probs = softmax(np.where(keep, scaled, -np.inf))
    rng = rng or np.random.default_rng()
    tok = int(rng.choice(len(probs), p=probs))
    info["final_probs"] = probs
    return tok, info


def generate(model, ids: list[int], max_new_tokens: int = 40, stop_id: int | None = None,
             temperature: float = 0.8, top_k: int | None = 20, top_p: float | None = 0.95,
             seed: int | None = 0, on_step=None) -> list[int]:
    """The autoregressive loop. Returns only the NEW token ids."""
    rng = np.random.default_rng(seed)
    ids, new = list(ids), []
    for _ in range(max_new_tokens):
        window = ids[-model.cfg.context_length:]           # the model can only see this many tokens
        logits = model.forward(np.array([window]))[0, -1]  # we only need the LAST position's prediction
        tok, info = sample_next(logits, temperature, top_k, top_p, rng)
        if on_step:
            on_step(tok, info)
        if tok == stop_id:
            break
        ids.append(tok)
        new.append(tok)
    return new
