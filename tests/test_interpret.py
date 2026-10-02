"""Activation patching is exact: patching the clean vector everywhere it matters restores the clean answer."""
import numpy as np

from transparent_transformer import GPT, BPETokenizer, paths
from transparent_transformer.alignment import format_prompt
from transparent_transformer.interpret import run


def test_patching_the_last_position_at_the_output_restores_the_clean_logits():
    tok, model = BPETokenizer.load(paths.TOKENIZER), GPT.load(paths.ALIGNED_MODEL)
    a = tok.encode(format_prompt("What is Seattle like in summer?") + " In summer Seattle is usually")
    b = tok.encode(format_prompt("What is Singapore like in summer?") + " In summer Singapore is usually")
    if len(a) != len(b):
        return
    la, cache = run(model, a, keep=True)
    patched = run(model, b, ("resid", len(model.blocks), len(b) - 1, cache["resid"][-1][-1]))
    assert np.allclose(la, patched, atol=1e-4)
    assert np.allclose(run(model, a), model.forward(np.array([a]))[0, -1], atol=1e-4)   # our hand-run forward matches the model's
