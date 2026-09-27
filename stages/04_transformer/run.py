"""Stage 4 - the full forward pass, with the shape of the data after every step.
Run:  python stages/04_transformer/run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

import numpy as np

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.alignment import format_prompt

tok = BPETokenizer.load(paths.TOKENIZER)
model = GPT.load(paths.ALIGNED_MODEL)
cfg = model.cfg

print("WHERE THE PARAMETERS LIVE")
groups: dict[str, int] = {}
for name, p in model.parameters().items():
    key = "embedding (token + position tables)" if name.startswith("embed") else           "attention (Q, K, V and output projections)" if ".attn." in name else           "MLP (Multi-Layer Perceptron)" if ".mlp." in name else "layer norms"
    groups[key] = groups.get(key, 0) + p.size
total = model.num_parameters()
for k, v in groups.items():
    print(f"   {k:<46}{v:>9,}  {'#' * int(40 * v / total)}")
print(f"   {'total':<46}{total:>9,}\n")

ids = np.array([tok.encode(format_prompt("What is the weather in Los Angeles?"))])
logits = model.forward(ids, capture=True)
cap = model.captured
print("THE FORWARD PASS  (B = batch, T = tokens, d = d_model, V = vocab_size)")
print(f"   token ids                     {ids.shape}            (B, T)")
print(f"   after embedding               {cap['stream'][0].shape}        (B, T, d)")
for i in range(cfg.n_layers):
    print(f"   block {i+1}: attention weights    {cap['attention'][i].shape}     (B, heads, T, T)")
    print(f"   block {i+1}: stream after block   {cap['stream'][i+1].shape}        (B, T, d)   <- same shape in, same shape out")
print(f"   logits                        {logits.shape}       (B, T, V)   one score per vocabulary entry, per position\n")

last = logits[0, -1]
print("the 5 highest-scoring next tokens after the prompt:")
for j in np.argsort(-last)[:5]:
    print(f"   {tok.token_str(j)!r:>10}  logit {last[j]:+.2f}")
print("\nLogits are raw scores, not probabilities yet. Stage 9 handles that.")
print("->  python stages/05_attention_closeup/run.py")
