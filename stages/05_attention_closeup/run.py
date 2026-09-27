"""Stage 5 - attention, computed by hand for one head, then drawn as a heat map.
Run:  python stages/05_attention_closeup/run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

import numpy as np

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.alignment import format_prompt

tok = BPETokenizer.load(paths.TOKENIZER)
model = GPT.load(paths.ALIGNED_MODEL)
ids = tok.encode(format_prompt("What is the weather in Los Angeles?"))
pieces = [tok.token_str(i) for i in ids]
model.forward(np.array([ids]), capture=True)

# ---- redo block 1, head 1 by hand, in six lines --------------------------------------------
blk = model.blocks[0]
x = blk.ln1.forward(model.captured["stream"][0])[0]            # (T, d)  normalised token vectors
hd = model.cfg.d_head
qkv = x @ blk.attn.qkv.params["W"] + blk.attn.qkv.params["b"]   # (T, 3d)
d = model.cfg.d_model
Q, K = qkv[:, 0:hd], qkv[:, d:d + hd]                           # head 1 = the first d_head columns of Q and of K
scores = Q @ K.T / np.sqrt(hd)                                  # how well does each key answer each query?
scores[np.triu(np.ones_like(scores, dtype=bool), k=1)] = -1e9   # causal mask: hide the future
weights = np.exp(scores - scores.max(-1, keepdims=True))
weights /= weights.sum(-1, keepdims=True)                       # softmax: each row sums to 1
assert np.allclose(weights, model.captured["attention"][0][0, 0], atol=1e-5)
print("recomputed block 1 / head 1 by hand. It matches the model exactly.\n")

shades = " ░▒▓█"
print("rows = the token doing the looking, columns = the token being looked at")
print("the empty upper-right triangle is the causal mask: no token can see the future\n")
for i, p in enumerate(pieces):
    row = "".join(shades[min(4, int(w * 4.999))] * 2 for w in weights[i, :i + 1])
    print(f"   {p.strip()[:12]:>13} |{row}")

print("\nwhat the final token reads from, per head (it is about to write the answer):")
for li, att in enumerate(model.captured["attention"]):
    for h in range(att.shape[1]):
        w = att[0, h, -1]
        top = np.argsort(-w)[:3]
        print(f"   block {li+1} head {h+1}: " + "   ".join(f"{pieces[j]!r} {w[j]:.0%}" for j in top))
print("\n->  python stages/06_pretraining/run.py")
