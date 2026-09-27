"""Stage 10 - the autoregressive loop, one line per generated token, then decoding back to text.
Run:  python stages/10_output/run.py "What is the weather in Los Angeles?" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

import numpy as np

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.alignment import format_prompt
from transparent_transformer.sampling import sample_next

tok = BPETokenizer.load(paths.TOKENIZER)
model = GPT.load(paths.ALIGNED_MODEL)
prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the weather in Los Angeles?"
ids = tok.encode(format_prompt(prompt))
rng, new = np.random.default_rng(0), []

print(f"{'pass':>4}  {'tokens in':>9}  {'chosen':<14}{'p':>7}   text so far")
for n in range(48):
    logits = model.forward(np.array([ids]))[0, -1]            # full forward pass over EVERYTHING so far
    t, info = sample_next(logits, temperature=0.7, rng=rng)
    print(f"{n+1:>4}  {len(ids):>9}  {tok.token_str(t)!r:<14}{info['final_probs'][t]:>7.1%}   {tok.decode(new)}")
    if t == tok.special["<|end|>"]:
        break
    ids.append(t)
    new.append(t)

print(f"\nthe model chose <|end|>, so generation stops. {len(new)} tokens took {len(new)+1} forward passes.")
print(f"\ndecode({new[:6]} ...)\n\n   \033[1;32m{tok.decode(new).strip()}\033[0m\n")
print("That is the entire trick. For the whole journey in one go:  python -m transparent_transformer.trace")
