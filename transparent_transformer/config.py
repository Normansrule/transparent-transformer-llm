"""All the knobs of the model in one place."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    vocab_size: int = 768       # how many different tokens exist
    context_length: int = 64    # how many tokens the model can look at at once
    d_model: int = 64           # width of every token vector (the "residual stream")
    n_heads: int = 4            # attention heads per block (d_model must divide evenly)
    n_layers: int = 2           # how many transformer blocks are stacked
    d_ff: int = 256             # hidden width of the Multi-Layer Perceptron (MLP), usually 4 x d_model
    seed: int = 1337

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


# Ready-made sizes. Bigger = knows more and copes better with messy questions, but trains slower.
# name: (model settings, pretraining steps, batch size, SFT steps, DPO steps)
PRESETS = {
    "tiny":   (dict(vocab_size=768,  context_length=64, d_model=64,  n_heads=4, n_layers=2, d_ff=256), 1500, 16, 300, 100),
    "small":  (dict(vocab_size=1024, context_length=96, d_model=96,  n_heads=4, n_layers=3, d_ff=384), 3000, 24, 1500, 150),
    "medium": (dict(vocab_size=1536, context_length=96, d_model=128, n_heads=4, n_layers=4, d_ff=512), 4000, 32, 2000, 150),
}
