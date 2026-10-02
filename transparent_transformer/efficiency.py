"""
MAKING IT CHEAPER: ADAPT WITH LoRA, SHRINK WITH QUANTIZATION
=============================================================
Run:  python -m transparent_transformer.efficiency          (about six minutes)

1. LoRA (Low-Rank Adaptation, Hu et al. 2021). To teach a model something new you can update all its weights
   (full fine-tuning), or freeze them and learn a small correction for each weight matrix:
        W x   becomes   W x + (alpha / r) * B A x          A: d_in x r,  B: r x d_out,  r = 4
   Only A and B learn: 8,192 numbers instead of 153,344. We teach six new cities both ways, with and without
   replaying old conversations, and measure what is learned and what is forgotten.

2. QUANTIZATION. Store each weight in fewer bits: 8 bits (256 levels) or 4 bits (16 levels), with one scale
   per output column, or one scale per group of 32 weights. Measure what the smaller file costs in quality.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np

from . import paths
from .alignment import chat, encode_example, honesty_rate, load_jsonl, pad_batch, safety_rates
from .loss import cross_entropy
from .optimizer import AdamW, clip_gradients, cosine_schedule
from .pretrain import get_batch
from .tokenizer import BPETokenizer
from .transformer import GPT

# Six nearby cities the model has never been taught. Short descriptions written for this lesson.
NEW = [("San Pedro", "mild and breezy", "mild and cloudy", "watch the ships"), ("Torrance", "warm and sunny", "mild and clear", "go to the beach"),
       ("Pasadena", "hot and dry", "cool and clear", "walk in the gardens"), ("Irvine", "warm and sunny", "mild and clear", "ride bikes"),
       ("Long Beach", "warm and breezy", "mild and clear", "go sailing"), ("Santa Monica", "mild and sunny", "cool and clear", "walk on the pier")]
TRAIN_TEMPLATES = ["What is {c} like in {s}?", "What is {s} like in {c}?"]
TEST_TEMPLATE = "How is {c} in {s}?"                                   # a phrasing never used for the new cities


def answer(c, s, summer, winter, act):
    return f"In summer {c} is usually {summer}. People there often {act}." if s == "summer" else f"In winter {c} is usually {winter}."


def new_rows():
    return [{"prompt": t.format(c=c, s=s), "response": answer(c, s, su, wi, act)} for c, su, wi, act in NEW for s in ("summer", "winter") for t in TRAIN_TEMPLATES]


def new_fact_accuracy(model, tok) -> float:
    ok = 0
    for c, su, wi, _ in NEW:
        for s, fact in (("summer", su), ("winter", wi)):
            a = chat(model, tok, TEST_TEMPLATE.format(c=c, s=s))
            ok += c in a and fact in a
    return ok / (2 * len(NEW))


def measure(model, tok) -> dict:
    from .agent_eval import CASES, passed
    from .harness import Harness
    h = Harness(offline=True, tricks=())
    h.model = model

    def ask(q):
        h.memory = []
        return h.reply(q)
    clean = [(q, s) for cat, q, s in CASES if cat == "clean climate question"]
    sr = safety_rates(model, tok)
    return {"new_facts": new_fact_accuracy(model, tok), "old_facts": sum(passed(ask(q), s) for q, s in clean) / len(clean),
            "refuses_harmful": sr["refuses_harmful"], "honest_sampled": honesty_rate(model, tok),
            "test_45": sum(passed(ask(q), s) for _, q, s in CASES) / len(CASES)}


# ------------------------------------------------------------------ LoRA, by hand
def attach_lora(model, r=4, alpha=8.0, seed=0):
    """Freeze every weight; give each attention and perceptron matrix a trainable low-rank correction B·A."""
    rng = np.random.default_rng(seed)
    params, grads, s = {}, {}, alpha / r
    for i, b in enumerate(model.blocks):
        for part, names in (("attn", ("qkv", "proj")), ("mlp", ("up", "down"))):
            for nm in names:
                lin = getattr(getattr(b, part), nm)
                d_in, d_out = lin.params["W"].shape
                key = f"block{i}.{part}.{nm}"
                A = (rng.standard_normal((d_in, r)) / np.sqrt(d_in)).astype(np.float32)
                B = np.zeros((r, d_out), np.float32)                     # starts at zero: the model is unchanged at step 0
                params[key + ".A"], params[key + ".B"] = A, B

                def fwd(x, lin=lin, A=A, B=B):
                    lin.x, lin.xa = x, x @ A
                    return x @ lin.params["W"] + lin.params["b"] + (lin.xa @ B) * s

                def bwd(dout, lin=lin, A=A, B=B, key=key, d_in=d_in):
                    x2, d2, xa2 = lin.x.reshape(-1, d_in), dout.reshape(-1, dout.shape[-1]), lin.xa.reshape(-1, A.shape[1])
                    grads[key + ".B"] = s * xa2.T @ d2                     # dL/dB
                    dxa = s * (dout @ B.T)                                 # dL/d(xA)
                    grads[key + ".A"] = x2.T @ dxa.reshape(-1, A.shape[1]) # dL/dA
                    return dout @ lin.params["W"].T + dxa @ A.T            # dL/dx through both paths
                lin.forward, lin.backward = fwd, bwd
    return params, grads


def merge_lora(model, params, r=4, alpha=8.0):
    """Fold B·A into W, so the adapted model runs exactly as fast as the original."""
    for i, b in enumerate(model.blocks):
        for part, names in (("attn", ("qkv", "proj")), ("mlp", ("up", "down"))):
            for nm in names:
                lin, key = getattr(getattr(b, part), nm), f"block{i}.{part}.{nm}"
                lin.params["W"] = lin.params["W"] + (alpha / r) * params[key + ".A"] @ params[key + ".B"]
                del lin.forward, lin.backward                              # back to the ordinary Linear methods


def finetune(tok, method: str, replay: bool, steps=300, bs=16):
    model = GPT.load(paths.ALIGNED_MODEL)
    pad, limit = tok.special["<|end|>"], model.cfg.context_length + 1
    new = [e for e in (encode_example(tok, r["prompt"], r["response"]) for r in new_rows()) if len(e[0]) <= limit]
    old = [e for e in (encode_example(tok, r["prompt"], r["response"]) for r in load_jsonl("sft.jsonl")) if len(e[0]) <= limit]
    if method == "lora":
        params, grads = attach_lora(model)
        lr = 3e-3
    else:
        params, grads, lr = None, None, 5e-5
    trainable = sum(p.size for p in params.values()) if params else model.num_parameters()
    opt = AdamW(params if params else model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    for step in range(steps):
        k = bs // 2 if replay else bs
        batch = [new[i] for i in rng.integers(0, len(new), k)] + ([old[i] for i in rng.integers(0, len(old), bs - k)] if replay else [])
        x, y, mask = pad_batch(batch, pad)
        loss, dlogits = cross_entropy(model.forward(x), y, mask)
        model.backward(dlogits)
        g = dict(grads) if params else model.gradients()
        clip_gradients(g, 1.0)
        opt.step(g, lr=cosine_schedule(step, steps, lr, warmup=20))
    if params:
        merge_lora(model, params)
    return model, trainable


# ------------------------------------------------------------------ quantization, by hand
def quantize(model, bits: int, group: int | None):
    """Round every weight matrix to 2^bits levels: one scale per output column, or per group of `group` inputs."""
    q, levels, stored = model.copy(), 2 ** (bits - 1) - 1, 0
    new = {}
    for k, w in q.parameters().items():
        if w.ndim != 2:
            new[k] = w; stored += w.size * 32
            continue
        if group:
            g = w.reshape(-1, group, w.shape[1]) if w.shape[0] % group == 0 else None
            if g is not None:
                sc = np.abs(g).max(axis=1, keepdims=True) / levels + 1e-12
                new[k] = (np.round(g / sc) * sc).reshape(w.shape).astype(np.float32)
                stored += w.size * bits + sc.size * 16
                continue
        sc = np.abs(w).max(axis=0, keepdims=True) / levels + 1e-12
        new[k] = (np.round(w / sc) * sc).astype(np.float32)
        stored += w.size * bits + sc.size * 16
    q.set_parameters(new)
    return q, stored / 8 / 1024


def val_loss(model, tok, data) -> float:
    x, y = get_batch(data, 64, model.cfg.context_length, np.random.default_rng(0))
    loss, _ = cross_entropy(model.forward(x), y, np.ones_like(y, dtype=np.float32))
    return float(loss)


def main() -> None:
    t0 = time.time()
    tok = BPETokenizer.load(paths.TOKENIZER)
    base = GPT.load(paths.ALIGNED_MODEL)
    before = measure(base, tok)
    print(f"before: {json.dumps({k: round(v, 2) for k, v in before.items()})}\n")
    print("1. LoRA versus full fine-tuning: teach six new cities")
    lora = {}
    for method in ("full", "lora"):
        for replay in (False, True):
            m, n = finetune(tok, method, replay)
            r = measure(m, tok)
            name = f"{'full fine-tune' if method == 'full' else 'LoRA, rank 4'}{', with replay' if replay else ''}"
            lora[name] = {"trainable": n, **r, "example": chat(m, tok, TEST_TEMPLATE.format(c="San Pedro", s="summer"))}
            print(f"   {name:<32} trains {n:>7,} numbers | " + "  ".join(f"{k} {v:.0%}" for k, v in r.items()) + f"   ({time.time() - t0:.0f}s)")
    print("\n2. quantization")
    data = np.array(tok.encode((paths.DATA / "pretrain.txt").read_text()), dtype=np.int64)
    val = data[int(len(data) * 0.95):]
    quant = {}
    for label, bits, group in (("32-bit (original)", 32, None), ("8-bit", 8, None), ("4-bit, one scale per column", 4, None), ("4-bit, groups of 32", 4, 32)):
        m, kb = (base, base.num_parameters() * 4 / 1024) if bits == 32 else quantize(base, bits, group)
        sr = safety_rates(m, tok)
        from .agent_eval import CASES, passed
        from .harness import Harness
        h = Harness(offline=True, tricks=()); h.model = m
        t45 = 0
        for _, q, s in CASES:
            h.memory = []; t45 += passed(h.reply(q), s)
        quant[label] = {"kb": kb, "val_loss": val_loss(m, tok, val), "test_45": t45 / len(CASES), "refuses_harmful": sr["refuses_harmful"],
                        "answer": chat(m, tok, "What is Los Angeles like in summer?")}
        print(f"   {label:<28} {kb:7.0f} KB | val loss {quant[label]['val_loss']:.3f} | test {t45 / len(CASES):.0%} | refuses {sr['refuses_harmful']:.0%} | {quant[label]['answer'][:60]}")
    (paths.ARTIFACTS / "efficiency_log.json").write_text(json.dumps({"before": before, "lora": lora, "quant": quant, "new_cities": NEW}, indent=1))
    # Text prediction must be measured on the BASE model: supervised fine-tuning on chat alone erased it (the alignment tax).
    bm = GPT.load(paths.ARTIFACTS / "base.npz")
    for label, bits, group in (("32-bit (original)", 32, None), ("8-bit", 8, None), ("4-bit, one scale per column", 4, None), ("4-bit, groups of 32", 4, 32)):
        quant[label]["val_loss_base"] = val_loss(bm if bits == 32 else quantize(bm, bits, group)[0], tok, val)
    tax = {n: val_loss(GPT.load(paths.ARTIFACTS / f), tok, val) for n, f in (("base", "base.npz"), ("sft", "sft.npz"), ("aligned", "aligned.npz"))}
    print(f"   base model's loss on held-out text: " + ", ".join(f"{k} {v['val_loss_base']:.3f}" for k, v in quant.items()))
    print(f"   the alignment tax: loss on ordinary text, base {tax['base']:.2f} -> after chat fine-tuning {tax['sft']:.2f}")
    (paths.ARTIFACTS / "efficiency_log.json").write_text(json.dumps({"before": before, "lora": lora, "quant": quant, "new_cities": NEW, "alignment_tax": tax}, indent=1))
    print(f"\nsaved artifacts/efficiency_log.json ({time.time() - t0:.0f}s). The shipped model is unchanged: this is an experiment.")


if __name__ == "__main__":
    main()
