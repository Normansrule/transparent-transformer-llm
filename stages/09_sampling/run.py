"""Stage 9 - one set of logits, many possible behaviours.
Run:  python stages/09_sampling/run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

import numpy as np

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.alignment import chat
from transparent_transformer.sampling import sample_next

tok = BPETokenizer.load(paths.TOKENIZER)
model = GPT.load(paths.SFT_MODEL)       # the SFT model is undecided between two answer styles: good for this demo
prompt = "<|user|>What is the weather in Seattle?<|assistant|>"
logits = model.forward(np.array([tok.encode(prompt)]))[0, -1]

print("the SAME logits, reshaped by temperature. Probability of the top 4 candidates:\n")
top = np.argsort(-logits)[:4]
print(f"   {'temperature':<14}" + "".join(f"{tok.token_str(j)!r:>12}" for j in top))
for T in [0.2, 0.7, 1.0, 1.5, 3.0]:
    _, info = sample_next(logits, temperature=T, top_k=None, top_p=None, rng=np.random.default_rng(0))
    print(f"   {T:<14}" + "".join(f"{info['final_probs'][j]:>12.1%}" for j in top))
print("\n   low temperature -> the favourite wins almost always.  high -> the underdogs get a chance.\n")

for T in [0.0, 0.8, 2.5]:
    print(f"temperature {T}: five answers to 'What is the weather in Seattle?'")
    for seed in range(5):
        print(f"   {chat(model, tok, 'What is the weather in Seattle?', temperature=T, top_k=None, top_p=None, seed=seed)}")
    print()
print("->  python stages/10_output/run.py")
