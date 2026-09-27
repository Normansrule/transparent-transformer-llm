"""
Proof that the hand-written backpropagation is correct.

For a handful of randomly chosen weights we estimate the gradient the slow,
obviously-correct way:

    (loss(w + h) - loss(w - h)) / 2h

and compare it to what model.backward() computed. If they agree to several
decimal places, every formula in layers.py / attention.py / transformer.py
must be right.
"""
import numpy as np

from transparent_transformer import GPT, Config
from transparent_transformer.loss import cross_entropy


def _tiny_model():
    cfg = Config(vocab_size=23, context_length=8, d_model=16, n_heads=4, n_layers=2, d_ff=32, seed=0)
    model = GPT(cfg)
    rng = np.random.default_rng(1)
    # float64 + larger weights so the numerical estimate is precise
    model.set_parameters({k: (v.astype(np.float64) + rng.standard_normal(v.shape) * 0.1)
                          for k, v in model.parameters().items()})
    return model, rng


def test_backprop_matches_numerical_gradient():
    model, rng = _tiny_model()
    ids = rng.integers(0, 23, size=(3, 8))
    targets = rng.integers(0, 23, size=(3, 8))
    mask = (rng.random((3, 8)) > 0.3).astype(np.float64)

    _, dlogits = cross_entropy(model.forward(ids), targets, mask)
    model.backward(dlogits)
    analytic = model.gradients()
    params = model.parameters()

    h, worst = 1e-5, 0.0
    for name, p in params.items():
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + h
            lp, _ = cross_entropy(model.forward(ids), targets, mask)
            p[idx] = old - h
            lm, _ = cross_entropy(model.forward(ids), targets, mask)
            p[idx] = old
            numeric = (lp - lm) / (2 * h)
            diff = abs(numeric - analytic[name][idx])
            err = diff / max(1e-12, abs(numeric) + abs(analytic[name][idx]))
            if abs(numeric) > 1e-6:
                worst = max(worst, err)
            assert err < 1e-4 or diff < 1e-9, (name, idx, numeric, analytic[name][idx])
    print(f"\nworst relative error across all checked weights: {worst:.2e}")


def test_causal_mask_blocks_the_future():
    """Changing a LATER token must never change the prediction at an EARLIER position."""
    model, rng = _tiny_model()
    a = rng.integers(0, 23, size=(1, 8))
    b = a.copy()
    b[0, 5:] = (b[0, 5:] + 1) % 23
    la, lb = model.forward(a), model.forward(b)
    assert np.allclose(la[0, :5], lb[0, :5])
    assert not np.allclose(la[0, 5:], lb[0, 5:])
