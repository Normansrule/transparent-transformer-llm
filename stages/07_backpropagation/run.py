"""Stage 7 - ONE training step in slow motion, on a fresh untrained model.
Run:  python stages/07_backpropagation/run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # so `import transparent_transformer` works from anywhere

import numpy as np

from transparent_transformer import GPT, BPETokenizer, Config, paths
from transparent_transformer.loss import cross_entropy

tok = BPETokenizer.load(paths.TOKENIZER)
model = GPT(Config())                                  # random weights: it knows nothing
ids = tok.encode("The weather in Los Angeles is usually sunny and warm.")
x, y = np.array([ids[:-1]]), np.array([ids[1:]])

print("1. FORWARD   run the sentence through the model")
loss, dlogits = cross_entropy(model.forward(x), y)
print(f"             loss = {loss:.4f}\n")

print("2. BACKWARD  start from dLoss/dlogits = probabilities - one_hot(correct), walk back layer by layer")
model.backward(dlogits)
grads = model.gradients()
print(f"             {'weight matrix':<22}{'shape':<14}{'size of its gradient'}")
for name in ["ln_f.g", "block1.mlp.down.W", "block1.attn.qkv.W", "block0.mlp.down.W", "block0.attn.qkv.W", "embed.tok"]:
    g = grads[name]
    print(f"             {name:<22}{str(g.shape):<14}{np.linalg.norm(g):.4f}")

print("\n3. CHECK     is the hand-written calculus right? Wiggle ONE weight and measure the loss directly")
name, idx, h = "block0.mlp.up.W", (3, 7), 1e-2
w = model.parameters()[name]
old = w[idx]
w[idx] = old + h; up, _ = cross_entropy(model.forward(x), y)
w[idx] = old - h; down, _ = cross_entropy(model.forward(x), y)
w[idx] = old
print(f"             wiggle estimate  (loss(w+h) - loss(w-h)) / 2h = {(up - down) / (2 * h):+.6f}")
print(f"             backpropagation                                = {grads[name][idx]:+.6f}")
print("             same answer. Backprop gets ALL 100,000+ gradients from one backward pass;")
print("             wiggling would need two forward passes PER WEIGHT.\n")

print("4. UPDATE    w = w - learning_rate * gradient, for every weight")
model.forward(x); model.backward(dlogits)              # restore the caches for a clean update
for k, p in model.parameters().items():
    p -= 0.5 * model.gradients()[k]
after, _ = cross_entropy(model.forward(x), y)
print(f"             loss before {loss:.4f}  ->  after {after:.4f}")
print("\nThat drop is learning. Pretraining is this step, repeated 1,500 times on different text.")
print("->  python stages/08_alignment/run.py")
