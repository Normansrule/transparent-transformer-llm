"""transparent-transformer-llm: a tiny Large Language Model (LLM) you can see all the way through.

Every module maps to one stage of the pipeline:

    tokenizer.py   -> stage 2   text        -> token ids
    embedding.py   -> stage 3   token ids   -> vectors
    transformer.py -> stage 4   vectors     -> next-token scores (logits)
    attention.py   -> stage 5   the close-up of what happens inside a block
    layers.py      -> stage 5   Linear, LayerNorm, GELU, MLP
    pretrain.py    -> stage 6   learning from raw text
    loss.py        -> stage 6/7 cross-entropy and where backprop starts
    optimizer.py   -> stage 7   AdamW: turning gradients into weight updates
    alignment.py   -> stage 8   SFT and DPO: turning a base model into an assistant
    sampling.py    -> stage 9   logits      -> one chosen token
    trace.py       -> all       run one prompt through every stage, record everything
"""
from .config import Config
from .tokenizer import BPETokenizer
from .transformer import GPT
from . import paths

__all__ = ["Config", "BPETokenizer", "GPT", "paths"]
