"""
LOOKING INSIDE - WHERE DOES THE MODEL KEEP WHAT IT KNOWS?
==========================================================
Run:  python -m transparent_transformer.interpret        (about two minutes)

Two standard interpretability techniques, on the real aligned model, gradients by hand:

1. CAUSAL TRACING (activation patching). Run the model twice: once on a sentence about Los Angeles (clean),
   once on the same sentence about another city (corrupted). Then re-run the corrupted sentence but paste ONE
   internal vector from the clean run in. If the clean answer comes back, that vector carried the fact.
   Score = how much of the clean-vs-corrupted logit difference returns (0% = none, 100% = all).

2. A SPARSE AUTOENCODER (dictionary learning). Individual neurons tend to respond to many unrelated things
   (polysemantic). An SAE re-expresses the 256 neuron activations of the block-2 perceptron layer as a
   combination of a few of 512 learned FEATURES, each of which ideally means one thing:
        f = ReLU((h - b_dec) W_enc + b_enc)     h_hat = f W_dec + b_dec
        loss = |h - h_hat|^2  +  lambda * sum(f)        (rows of W_dec kept at length 1)
   This is the method of Anthropic's "Towards Monosemanticity" (2023) and OpenAI's "Extracting Concepts from
   GPT-4" (2024), at toy scale. We then compare how single-minded features are versus raw neurons.

Writes artifacts/interpret_log.json and docs/inside.js (for the website's feature browser).
"""
from __future__ import annotations

import base64
import re
import json
import sys
import time
from collections import Counter

import numpy as np

from . import paths
from .alignment import format_prompt, load_jsonl
from .tokenizer import BPETokenizer
from .transformer import GPT

sys.path.insert(0, str(paths.ROOT / "data"))
from make_corpus import CITIES, HELD_OUT  # noqa: E402

SAE_LAYER = 1                      # second block's perceptron layer


# ------------------------------------------------------------------ a forward pass we can reach into
def run(model, ids, patch=None, keep=False):
    """patch = (kind, layer, pos, vector), kind in {"resid", "attn", "mlp"}. Returns last-position logits (and the cache)."""
    x = model.embed.forward(np.array([ids]))
    cache = {"resid": [], "attn": [], "mlp": [], "hidden": []}
    for i, b in enumerate(model.blocks):
        if patch and patch[0] == "resid" and patch[1] == i:
            x = x.copy(); x[0, patch[2]] = patch[3]
        cache["resid"].append(x[0].copy())
        a = b.attn.forward(b.ln1.forward(x))
        if patch and patch[0] == "attn" and patch[1] == i:
            a = a.copy(); a[0, patch[2]] = patch[3]
        x = x + a
        hid = b.mlp.act.forward(b.mlp.up.forward(b.ln2.forward(x)))
        m = b.mlp.down.forward(hid)
        if patch and patch[0] == "mlp" and patch[1] == i:
            m = m.copy(); m[0, patch[2]] = patch[3]
        x = x + m
        cache["attn"].append(a[0].copy()); cache["mlp"].append(m[0].copy()); cache["hidden"].append(hid[0].copy())
    if patch and patch[0] == "resid" and patch[1] == len(model.blocks):
        x = x.copy(); x[0, patch[2]] = patch[3]
    cache["resid"].append(x[0].copy())
    logits = model.ln_f.forward(x)[0, -1] @ model.embed.params["tok"].T
    return (logits, cache) if keep else logits


# ------------------------------------------------------------------ 1. causal tracing
def city_pairs(model, tok):
    """Pairs of cities whose sentences tokenize to the same length and whose summer facts start differently."""
    known = [c for c in CITIES if c[0] not in HELD_OUT]
    text = lambda c: tok.encode(format_prompt(f"What is {c} like in summer?") + f" In summer {c} is usually")   # noqa: E731
    first = {c[0]: tok.encode(" " + c[3])[0] for c in known}
    good = [c[0] for c in known if int(np.argmax(run(model, text(c[0])))) == first[c[0]]]       # model gets it right
    pairs = []
    for a in good:
        for b in good:
            if a < b and len(text(a)) == len(text(b)) and first[a] != first[b]:
                pairs.append((a, b))
    return pairs, text, first


def trace(model, tok, max_pairs=12):
    pairs, text, first = city_pairs(model, tok)
    lengths = Counter(len(text(a)) for a, _ in pairs)
    common = lengths.most_common(1)[0][0]                      # same length -> every position means the same thing
    pairs = [p for p in pairs if len(text(p[0])) == common][:max_pairs]
    L, results = len(model.blocks), []
    for a, b in pairs:
        ia, ib, ta, tb = text(a), text(b), first[a], first[b]
        la, ca = run(model, ia, keep=True)
        lb = run(model, ib)
        d_clean, d_corr = la[ta] - la[tb], lb[ta] - lb[tb]
        rec = lambda lg: float((lg[ta] - lg[tb] - d_corr) / (d_clean - d_corr + 1e-9))          # noqa: E731
        T = len(ib)
        resid = [[rec(run(model, ib, ("resid", l, t, ca["resid"][l][t]))) for t in range(T)] for l in range(L + 1)]
        comp = {k: [[rec(run(model, ib, (k, l, t, ca[k][l][t]))) for t in range(T)] for l in range(L)] for k in ("attn", "mlp")}
        results.append({"clean": a, "corrupt": b, "pieces": [tok.token_str(i) for i in ib], "resid": resid, **comp,
                        "clean_word": tok.token_str(ta), "corrupt_word": tok.token_str(tb)})
    avg = {k: np.mean([r[k] for r in results], axis=0).round(3).tolist() for k in ("resid", "attn", "mlp")}
    return {"pairs": results, "average": avg, "pieces_template": results[0]["pieces"] if results else []}


# ------------------------------------------------------------------ 2. the sparse autoencoder
def collect(model, tok, rows):
    H, ctx = [], []
    for r in rows:
        ids = tok.encode(format_prompt(r["prompt"]) + " " + r["response"])[-model.cfg.context_length:]
        _, cache = run(model, ids, keep=True)
        H.append(cache["hidden"][SAE_LAYER])
        pieces = [tok.token_str(i) for i in ids]
        ctx += [(pieces, t) for t in range(len(ids))]
    return np.vstack(H).astype(np.float32), ctx


def train_sae(H, m=512, lam=3.0, steps=3000, batch=1024, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed)
    d = H.shape[1]
    W_dec = rng.normal(size=(m, d)).astype(np.float32); W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_enc, b_enc, b_dec = W_dec.T.copy(), np.zeros(m, np.float32), H.mean(0)
    params = [W_enc, b_enc, W_dec, b_dec]
    mom, vel, t = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params], 0
    for step in range(steps):
        x = H[rng.integers(0, len(H), batch)]
        pre = (x - b_dec) @ W_enc + b_enc
        f = np.maximum(pre, 0)
        xh = f @ W_dec + b_dec
        err = xh - x
        # ---- gradients, by hand: loss = mean |err|^2 + lam * mean sum f
        g_xh = 2 * err / batch
        g_Wdec = f.T @ g_xh
        g_bdec_out = g_xh.sum(0)
        g_f = g_xh @ W_dec.T + lam / batch
        g_pre = g_f * (pre > 0)
        g_Wenc = (x - b_dec).T @ g_pre
        g_benc = g_pre.sum(0)
        g_bdec = g_bdec_out - (g_pre @ W_enc.T).sum(0)
        t += 1
        for p, g, mo, ve in zip(params, [g_Wenc, g_benc, g_Wdec, g_bdec], mom, vel):
            mo *= 0.9; mo += 0.1 * g; ve *= 0.999; ve += 0.001 * g * g
            p -= lr * (mo / (1 - 0.9 ** t)) / (np.sqrt(ve / (1 - 0.999 ** t)) + 1e-8)
        W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)       # keep every feature direction at length 1
    pre = (H - b_dec) @ W_enc + b_enc
    F = np.maximum(pre, 0)
    recon = F @ W_dec + b_dec
    ev = 1 - ((recon - H) ** 2).sum() / ((H - H.mean(0)) ** 2).sum()
    return {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec}, F, float(ev), float((F > 0).sum(1).mean())


def word(pieces, t):
    """The whole word containing token t (pieces may split a word)."""
    s, e = t, t
    while s > 0 and not pieces[s].startswith((" ", "<")) and pieces[s - 1][-1:].isalpha():
        s -= 1
    while e + 1 < len(pieces) and not pieces[e + 1].startswith((" ", "<")) and pieces[e + 1][:1].isalpha():
        e += 1
    return "".join(pieces[s:e + 1]).strip()


def describe(acts, ctx, top=20):
    """For each unit: its top-activating contexts, the word it fires on most, and how single-minded it is."""
    out = []
    for j in range(acts.shape[1]):
        col = acts[:, j]
        live = int((col > 0).sum()) if col.min() >= 0 else int((col > 0.05).sum())
        idx = np.argsort(-col)[:top]
        words = [word(*ctx[i]) for i in idx]
        common, n = Counter(words).most_common(1)[0]
        ex = [{"before": "".join(ctx[i][0][max(0, ctx[i][1] - 6):ctx[i][1]]), "token": ctx[i][0][ctx[i][1]],
               "after": "".join(ctx[i][0][ctx[i][1] + 1:ctx[i][1] + 3]), "act": round(float(col[i]), 2)} for i in idx[:6]]
        out.append({"label": common, "purity": n / top, "freq": live / len(col), "examples": ex, "top_words": Counter(words).most_common(4)})
    return out


def concept_features(feats) -> dict:
    """Features whose top activations are 90%+ one CATEGORY of word (but at least 3 different words): concepts, not tokens."""
    from harmless import SKIES
    cities = {w for c in CITIES for w in c[0].split()}
    skip = {"What", "Right", "No", "In", "I", "People", "Live", "Tell", "It", "Is", "Hello", "Hi", "Thanks", "Say", "Write", "Help",
            "Pretend", "Create", "The", "Summer", "Winter"}
    cats = {"city names": lambda w: w.strip(".,?:") in cities or (w[:1].isupper() and w.strip(".,?:") not in skip),
            "numbers": lambda w: bool(re.fullmatch(r"\d+[,.]?", w)),
            "sky words": lambda w: w.strip(".,") in SKIES}
    out = {}
    for name, fn in cats.items():
        out[name] = []
        for i, f in enumerate(feats):
            ws = [w for w, n in f["top_words"] for _ in range(n)]
            if ws and f["freq"] > 0 and sum(fn(w) for w in ws) / len(ws) >= 0.9 and len(f["top_words"]) >= 3:
                out[name].append({"id": i, "words": [w for w, _ in f["top_words"]]})
    return out


def f16(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float16).tobytes()).decode()


def main() -> None:
    t0 = time.time()
    tok = BPETokenizer.load(paths.TOKENIZER)
    model = GPT.load(paths.ALIGNED_MODEL)
    print("1. causal tracing")
    tr = trace(model, tok)
    print(f"   {len(tr['pairs'])} city pairs, e.g. {tr['pairs'][0]['clean']} ('{tr['pairs'][0]['clean_word']}') vs "
          f"{tr['pairs'][0]['corrupt']} ('{tr['pairs'][0]['corrupt_word']}')")
    for l, row in enumerate(tr["average"]["resid"]):
        print(f"   stream entering {'block ' + str(l + 1) if l < len(model.blocks) else 'the output'}: " + " ".join(f"{v:+.2f}" for v in row))
    print(f"\n2. sparse autoencoder on block {SAE_LAYER + 1}'s perceptron layer")
    rows = load_jsonl("sft.jsonl")
    H, ctx = collect(model, tok, rows)
    scale = float(np.sqrt((H ** 2).sum(1).mean() / H.shape[1]))
    Hs = H / scale
    print(f"   {len(H):,} token activations of 256 neurons")
    sweep = []
    for lam in (0.3, 1.0, 3.0, 8.0):                         # the core trade-off: sparser features explain less
        _, Fx, evx, l0x = train_sae(Hs, lam=lam, steps=1200)
        sweep.append({"lambda": lam, "explained": evx, "l0": l0x, "dead": int((Fx.max(0) == 0).sum())})
        print(f"   sparsity penalty {lam:<4}: explains {evx:.0%} with {l0x:5.1f} active features per token, {sweep[-1]['dead']} never fire")
    sae, F, ev, l0 = train_sae(Hs)
    print(f"   512 features: explains {ev:.0%} of the variance with {l0:.1f} active features per token on average")
    feats = describe(F, ctx)
    neurons = describe(H, ctx)
    live = [i for i, f in enumerate(feats) if f["freq"] > 0]
    fp, npur = np.mean([feats[i]["purity"] for i in live]), np.mean([n["purity"] for n in neurons])
    print(f"   single-mindedness (share of a unit's top-20 activations on its most common word): "
          f"neurons {npur:.0%}, SAE features {fp:.0%}   ({len(feats) - len(live)} features never fire)")
    best = sorted(live, key=lambda i: (-feats[i]["purity"], -feats[i]["freq"]))[:8]
    for i in best:
        print(f"   feature {i:>3}: '{feats[i]['label']}' ({feats[i]['purity']:.0%} of top-20)   e.g. ...{feats[i]['examples'][0]['before'][-20:]}[{feats[i]['examples'][0]['token']}]")
    concepts = concept_features(feats)
    print("   concept features: " + ", ".join(f"{len(v)} for {k}" for k, v in concepts.items()))
    log = {"concept_features": concepts, "trace": tr, "sae": {"layer": SAE_LAYER, "features": 512, "explained_variance": ev, "l0": l0, "tokens": len(H),
                                "purity_neurons": npur, "purity_features": fp, "dead": len(feats) - len(live), "sweep": sweep},
           "best": best, "features": feats, "neuron_purity": [n["purity"] for n in neurons]}
    (paths.ARTIFACTS / "interpret_log.json").write_text(json.dumps(log))
    web = {"layer": SAE_LAYER, "scale": scale, "m": 512, "d": 256, "W_enc": f16(sae["W_enc"]), "b_enc": f16(sae["b_enc"]), "b_dec": f16(sae["b_dec"]),
           "features": [{"label": f["label"], "purity": round(f["purity"], 2), "freq": round(f["freq"], 4), "examples": f["examples"]} for f in feats],
           "trace": {"average": tr["average"], "pieces": tr["pieces_template"], "pairs": [{k: p[k] for k in ("clean", "corrupt", "clean_word", "corrupt_word", "resid", "attn", "mlp", "pieces")} for p in tr["pairs"]]},
           "purity_neurons": npur, "purity_features": fp, "explained_variance": ev, "l0": l0, "sweep": sweep, "concepts": concepts}
    (paths.DOCS / "inside.js").write_text("window.INSIDE = " + json.dumps(web) + ";\n")
    print(f"\nsaved artifacts/interpret_log.json and docs/inside.js ({(paths.DOCS / 'inside.js').stat().st_size / 1e6:.2f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
