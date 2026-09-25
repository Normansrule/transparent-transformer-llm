"""
STAGE 4 - THE TRANSFORMER
=========================
The whole model, top to bottom:

    token ids (B, T)
        |  Embedding                                   stage 3
        v
    x (B, T, d_model)  <- the "residual stream"
        |
        |  Block 1:   x = x + Attention(LayerNorm(x))  tokens exchange information
        |             x = x + MLP(LayerNorm(x))        each token is processed alone
        |  Block 2:   ... same again, new weights ...
        v
    LayerNorm
        |  multiply by the embedding table, transposed ("weight tying")
        v
    logits (B, T, vocab_size)   one score per possible next token, at every position

Notice the pattern  x = x + something(x).  Each block only ADDS a correction
to the stream; it never overwrites it. That "residual" design is what lets
gradients flow backwards through deep stacks without fading away.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .attention import CausalSelfAttention
from .config import Config
from .embedding import Embedding
from .layers import MLP, Layer, LayerNorm


class Block(Layer):
    """One transformer block = attention (mix between tokens) + MLP (think per token)."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        super().__init__()
        out_std = 0.02 / np.sqrt(2 * cfg.n_layers)       # GPT-2 trick: shrink layers that write to the stream
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, rng, out_std)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, rng, out_std)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.attn_out = self.attn.forward(self.ln1.forward(x))
        x = x + self.attn_out                            # residual connection 1
        self.mlp_out = self.mlp.forward(self.ln2.forward(x))
        return x + self.mlp_out                          # residual connection 2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # a "+" in the forward pass copies the gradient to both branches in the backward pass
        dx = dout + self.ln2.backward(self.mlp.backward(dout))
        return dx + self.ln1.backward(self.attn.backward(dx))


class GPT:
    """A Generative Pre-trained Transformer (GPT), small enough to read in one sitting."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.embed = Embedding(cfg.vocab_size, cfg.context_length, cfg.d_model, rng)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layers)]
        self.ln_f = LayerNorm(cfg.d_model)
        self.captured: dict = {}

    # ---------------------------------------------------------------- forward
    def forward(self, ids: np.ndarray, capture: bool = False) -> np.ndarray:
        """ids (B, T) -> logits (B, T, vocab_size). `capture=True` keeps every intermediate."""
        assert ids.shape[1] <= self.cfg.context_length, "prompt longer than context_length"
        x = self.embed.forward(ids)
        stream = [x]
        for block in self.blocks:
            x = block.forward(x)
            stream.append(x)
        self.h = self.ln_f.forward(x)
        logits = self.h @ self.embed.params["tok"].T     # weight tying: reuse the embedding table
        if capture:
            self.captured = {
                "token_vectors": self.embed.tok_vectors,
                "position_vectors": self.embed.pos_vectors,
                "stream": stream,
                "attention": [b.attn.weights for b in self.blocks],
                "attn_out": [b.attn_out for b in self.blocks],
                "mlp_out": [b.mlp_out for b in self.blocks],
                "final": self.h,
            }
        return logits

    # --------------------------------------------------------------- backward
    def backward(self, dlogits: np.ndarray) -> None:
        """Given dLoss/dlogits, fill in `.grads` for every weight in the model."""
        W = self.embed.params["tok"]
        V = W.shape[0]
        dW_head = dlogits.reshape(-1, V).T @ self.h.reshape(-1, W.shape[1])
        dx = self.ln_f.backward(dlogits @ W)
        for block in reversed(self.blocks):              # walk the blocks in REVERSE order
            dx = block.backward(dx)
        self.embed.backward(dx)
        self.embed.grads["tok"] += dW_head               # the tied table is used twice, so it gets two gradients

    # --------------------------------------------------- parameter bookkeeping
    def _named_layers(self):
        yield "embed", self.embed
        for i, b in enumerate(self.blocks):
            yield f"block{i}.ln1", b.ln1
            yield f"block{i}.attn.qkv", b.attn.qkv
            yield f"block{i}.attn.proj", b.attn.proj
            yield f"block{i}.ln2", b.ln2
            yield f"block{i}.mlp.up", b.mlp.up
            yield f"block{i}.mlp.down", b.mlp.down
        yield "ln_f", self.ln_f

    def parameters(self) -> dict[str, np.ndarray]:
        return {f"{n}.{k}": v for n, layer in self._named_layers() for k, v in layer.params.items()}

    def gradients(self) -> dict[str, np.ndarray]:
        return {f"{n}.{k}": v for n, layer in self._named_layers() for k, v in layer.grads.items()}

    def set_parameters(self, new: dict[str, np.ndarray]) -> None:
        for n, layer in self._named_layers():
            for k in layer.params:
                layer.params[k] = new[f"{n}.{k}"]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters().values())

    def copy(self) -> "GPT":
        other = GPT(self.cfg)
        other.set_parameters({k: v.copy() for k, v in self.parameters().items()})
        return other

    # ------------------------------------------------------------------- disk
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, __config__=json.dumps(self.cfg.to_dict()), **self.parameters())

    @classmethod
    def load(cls, path: str | Path) -> "GPT":
        data = np.load(path, allow_pickle=False)
        model = cls(Config(**json.loads(str(data["__config__"]))))
        model.set_parameters({k: data[k] for k in data.files if k != "__config__"})
        return model
