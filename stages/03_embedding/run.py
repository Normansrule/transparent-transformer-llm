"""Stage 3 - token ids -> vectors, and proof that the vectors learned MEANING.
Run:  python stages/03_embedding/run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

import numpy as np

from transparent_transformer import GPT, BPETokenizer, paths

tok = BPETokenizer.load(paths.TOKENIZER)
model = GPT.load(paths.ALIGNED_MODEL)
table = model.embed.params["tok"]

ids = tok.encode("What is the weather in Los Angeles?")
x = model.embed.forward(np.array([ids]))
print(f"embedding table : {table.shape}  = (vocab_size, d_model)   {table.size:,} learned numbers")
print(f"input           : {np.array(ids).shape} integers")
print(f"output          : {x.shape[1:]} floats\n")
for i in ids[:4]:
    print(f"   {tok.token_str(i)!r:>10} (id {i:>3}) -> [{' '.join(f'{v:+.2f}' for v in table[i, :8])} ...]")

print("\nNobody told the model what words mean. After training, tokens used in similar")
print("places have ended up with similar vectors. Nearest neighbours by cosine similarity:\n")
unit = table / np.linalg.norm(table, axis=1, keepdims=True)
for word in [" sunny", " cold", " winter", " Tokyo", " is"]:
    wid = tok.encode(word)
    if len(wid) != 1:
        continue
    sims = unit @ unit[wid[0]]
    best = [j for j in np.argsort(-sims) if j != wid[0] and len(tok.token_str(j).strip()) > 2][:5]
    print(f"   {word!r:>10} ~ " + ", ".join(f"{tok.token_str(j)!r} ({sims[j]:.2f})" for j in best))
print("\n->  python stages/04_transformer/run.py")
