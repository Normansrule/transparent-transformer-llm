"""Stage 6 - what one pretraining example looks like, and how well the finished base model plays the game.
Run:  python stages/06_pretraining/run.py          (to actually train:  python -m transparent_transformer.pretrain)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

import numpy as np

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.attention import softmax

tok = BPETokenizer.load(paths.TOKENIZER)
base = GPT.load(paths.BASE_MODEL)
fresh = GPT(base.cfg)                                  # same architecture, random weights

sentence = "The weather in Los Angeles is usually sunny and warm."
ids = tok.encode(sentence)
x, y = ids[:-1], ids[1:]
print("ONE TRAINING EXAMPLE = a piece of text and the same text shifted by one\n")
print(f"   {'the model sees':<34}{'it must predict':<18}{'untrained':>10}{'trained':>10}")
p_new = softmax(fresh.forward(np.array([x]))[0])
p_old = softmax(base.forward(np.array([x]))[0])
for t in range(len(x)):
    seen = tok.decode(x[max(0, t - 3):t + 1])
    print(f"   {('...' if t > 3 else '') + seen!r:<34}{tok.token_str(y[t])!r:<18}{p_new[t, y[t]]:>10.1%}{p_old[t, y[t]]:>10.1%}")
print("\n   (the two right-hand columns: probability each model gave to the CORRECT next token)")
print(f"\n   untrained loss: {-np.log(p_new[range(len(y)), y]).mean():.2f}    (ln({base.cfg.vocab_size}) = {np.log(base.cfg.vocab_size):.2f} is pure guessing)")
print(f"   trained loss  : {-np.log(p_old[range(len(y)), y]).mean():.2f}")
print("\nLook at ' sunny': the model had to learn a FACT about Los Angeles to win that round.")
print("->  python stages/07_backpropagation/run.py")
