"""LoRA by hand: invisible at the start, gradients correct, and merging changes nothing."""
import numpy as np

from transparent_transformer import GPT, paths
from transparent_transformer.efficiency import attach_lora, merge_lora, quantize
from transparent_transformer.loss import cross_entropy


def test_lora_gradient_and_merge():
    m = GPT.load(paths.ALIGNED_MODEL)
    x, y = np.array([[1, 2, 3, 4, 5]]), np.array([[2, 3, 4, 5, 6]])
    y0 = m.forward(x).copy()
    p, g = attach_lora(m)
    assert np.allclose(m.forward(x), y0, atol=1e-5)                    # B starts at zero: nothing changes
    for k in p:
        p[k] += np.random.default_rng(1).normal(0, 0.05, p[k].shape).astype(np.float32)
    loss = lambda: cross_entropy(m.forward(x), y)[0]                    # noqa: E731
    _, d = cross_entropy(m.forward(x), y); m.backward(d)
    k, i, e = "block0.attn.qkv.B", (2, 7), 1e-3
    an = g[k][i]; p[k][i] += e; l1 = loss(); p[k][i] -= 2 * e; l2 = loss(); p[k][i] += e
    assert abs(an - (l1 - l2) / (2 * e)) < 1e-3
    before = m.forward(x).copy(); merge_lora(m, p)
    assert np.allclose(m.forward(x), before, atol=1e-4)                # folding B·A into W gives the same outputs


def test_8bit_quantization_is_nearly_lossless():
    m = GPT.load(paths.ALIGNED_MODEL)
    q, kb = quantize(m, 8, None)
    x = np.array([[1, 2, 3, 4, 5]])
    assert kb < m.num_parameters() * 4 / 1024 / 3.5
    assert np.abs(q.forward(x) - m.forward(x)).max() < 0.5
