"""
SIDE TRIP - THE MULTI-LAYER PERCEPTRON, the classic 3Blue1Brown picture, trained for real.
Run:  pip install pillow && python perceptron/train_digits.py        (about a minute)

    784 inputs (a 28 x 28 image)  ->  16 neurons  ->  16 neurons  ->  10 outputs (one per digit)

The same shape of network as the famous video, trained here with the same hand-written backpropagation
style as the transformer. Handwritten digit datasets need downloading, so we MAKE our own: digits from the
fonts installed on this computer, randomly rotated, slanted, thickened and shifted, then centred the way the
classic MNIST dataset is (fit a 20 x 20 box, centre of mass in the middle of 28 x 28). The website applies the
exact same centring to whatever you draw.

Writes docs/mlp.js: the trained weights plus a few hundred test digits for the demo.
"""
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
prng = random.Random(0)

FONTS = [f for f in sorted(set(glob.glob("/usr/share/fonts/truetype/**/*.ttf", recursive=True)))
         if not any(bad in f.lower() for bad in ("math", "symbol", "emoji", "japanese", "ipa", "wqy", "noto"))]


def centre_like_mnist(img: np.ndarray) -> np.ndarray:
    """img: float array, ink = 1. Crop to the ink, scale the long side to 20 pixels, centre of mass to the middle."""
    ys, xs = np.nonzero(img > 0.1)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20 / max(h, w)
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS), np.float32) / 255
    out = np.zeros((28, 28), np.float32)
    h, w = small.shape
    y0, x0 = (28 - h) // 2, (28 - w) // 2
    out[y0:y0 + h, x0:x0 + w] = small
    tot = out.sum()
    cy, cx = (np.arange(28)[:, None] * out).sum() / tot, (np.arange(28)[None, :] * out).sum() / tot
    return np.roll(np.roll(out, int(round(13.5 - cy)), 0), int(round(13.5 - cx)), 1)


def render(digit: int) -> np.ndarray:
    font = ImageFont.truetype(prng.choice(FONTS), 64)
    im = Image.new("L", (112, 112), 0)
    ImageDraw.Draw(im).text((28, 12), str(digit), fill=255, font=font)
    im = im.filter(ImageFilter.MaxFilter(prng.choice([1, 3, 3, 5, 7])))          # stroke thickness
    shear = prng.uniform(-0.35, 0.35)                                            # slant
    im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * 56, 0, 1, 0), Image.BILINEAR)
    im = im.rotate(prng.uniform(-18, 18), Image.BILINEAR)
    sx, sy = prng.uniform(0.75, 1.15), prng.uniform(0.85, 1.15)                  # squash and stretch
    im = im.resize((int(112 * sx), int(112 * sy)), Image.BILINEAR)
    return centre_like_mnist(np.asarray(im, np.float32) / 255)


def make(n):
    X = np.zeros((n, 784), np.float32)
    y = np.zeros(n, np.int64)
    for i in range(n):
        y[i] = i % 10
        X[i] = render(int(y[i])).reshape(-1)
    return X, y


sig = lambda z: 1 / (1 + np.exp(-z))                                            # noqa: E731


def train(X, y, Xt, yt, epochs=40, sizes=(784, 16, 16, 10), lr=3e-3):
    Ws = [rng.standard_normal((a, b)).astype(np.float32) * math.sqrt(1 / a) for a, b in zip(sizes, sizes[1:])]
    bs = [np.zeros(b, np.float32) for b in sizes[1:]]
    m = [np.zeros_like(p) for p in Ws + bs]; v = [np.zeros_like(p) for p in Ws + bs]; t = 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        for k in range(0, len(X), 64):
            idx = order[k:k + 64]
            a = [X[idx]]
            for i, (W, b) in enumerate(zip(Ws, bs)):                             # forward: sigmoid hidden layers
                z = a[-1] @ W + b
                a.append(sig(z) if i < len(Ws) - 1 else z)
            p = np.exp(a[-1] - a[-1].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            d = p; d[np.arange(len(idx)), y[idx]] -= 1; d /= len(idx)            # softmax + cross-entropy
            gW, gb = [None] * len(Ws), [None] * len(Ws)
            for i in reversed(range(len(Ws))):                                   # backward: the chain rule, by hand
                gW[i], gb[i] = a[i].T @ d, d.sum(0)
                if i:
                    d = (d @ Ws[i].T) * a[i] * (1 - a[i])
            t += 1
            for j, (P, G) in enumerate(zip(Ws + bs, gW + gb)):                   # Adam
                m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G * G
                P -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"  epoch {ep + 1:>2}: test accuracy {accuracy(Ws, bs, Xt, yt):.1%}")
    return Ws, bs


def accuracy(Ws, bs, X, y):
    a = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = a @ W + b
        a = sig(a) if i < len(Ws) - 1 else a
    return float((a.argmax(1) == y).mean())


if __name__ == "__main__":
    print(f"rendering digits from {len(FONTS)} fonts ...")
    X, y = make(30000)
    Xt, yt = make(2000)
    print("training 784 -> 16 -> 16 -> 10 (sigmoid, like the video) ...")
    Ws, bs = train(X, y, Xt, yt)
    acc = accuracy(Ws, bs, Xt, yt)
    demo = [(Xt[i].round(2).tolist(), int(yt[i])) for i in range(300)]
    out = {"sizes": [784, 16, 16, 10], "activation": "sigmoid", "test_accuracy": acc,
           "W": [W.round(4).tolist() for W in Ws], "b": [b.round(4).tolist() for b in bs], "demo": demo}
    (ROOT / "docs" / "mlp.js").write_text("window.MLP = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"test accuracy {acc:.1%}  ->  wrote docs/mlp.js ({(ROOT / 'docs' / 'mlp.js').stat().st_size / 1e6:.1f} MB)")
