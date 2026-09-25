"""
THE WHOLE JOURNEY - ONE PROMPT, EVERY STAGE, NOTHING HIDDEN
===========================================================
Run:  python -m transparent_transformer.trace "What is the weather in Los Angeles?"

Follows a single prompt through all ten stages, printing the actual numbers at
each step, and saves everything to docs/trace.js so the interactive page
(docs/index.html) can replay it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import paths
from .alignment import chat, format_prompt
from .attention import softmax
from .sampling import sample_next
from .tokenizer import BPETokenizer
from .transformer import GPT

C = {"dim": "\033[2m", "b": "\033[1m", "amber": "\033[38;5;214m", "cyan": "\033[38;5;80m",
     "coral": "\033[38;5;210m", "mint": "\033[38;5;121m", "blue": "\033[38;5;75m", "x": "\033[0m"}
CHIP_COLOURS = [214, 80, 210, 121, 75, 183]


def chip(text: str, i: int) -> str:
    shown = text.replace("\n", "\\n")
    return f"\033[48;5;{CHIP_COLOURS[i % len(CHIP_COLOURS)]}m\033[30m{shown}\033[0m"


def bar(p: float, width: int = 28) -> str:
    return "█" * max(1, int(round(p * width))) if p > 0.004 else "·"


class Narrator:
    def __init__(self, animate: bool):
        self.animate = animate and sys.stdout.isatty()

    def header(self, n: int, title: str, where: str) -> None:
        if n > 1:
            self.conveyor()
        print(f"\n{C['b']}{C['amber']}┏━ STAGE {n:>2} ━ {title} {'━' * max(2, 58 - len(title))}┓{C['x']}")
        print(f"{C['dim']}   code: {where}{C['x']}\n")

    def conveyor(self) -> None:
        """The little machine that carries the data to the next stage."""
        if not self.animate:
            print(f"{C['dim']}        │\n        ▼{C['x']}")
            return
        for frame in range(14):
            belt = "".join("▰" if (i - frame) % 4 == 0 else "▱" for i in range(40))
            print(f"\r   {C['amber']}{belt}{C['x']}", end="", flush=True)
            time.sleep(0.04)
        print(f"\r{' ' * 46}\r{C['dim']}        ▼{C['x']}")

    def pause(self, s: float = 0.25) -> None:
        if self.animate:
            time.sleep(s)


def run(prompt: str, animate: bool = True, temperature: float = 0.7, seed: int = 0, quiet: bool = False) -> dict:
    tok = BPETokenizer.load(paths.TOKENIZER)
    model = GPT.load(paths.ALIGNED_MODEL)
    base = GPT.load(paths.BASE_MODEL)
    cfg = model.cfg
    say = Narrator(animate and not quiet)
    if quiet:
        import builtins
        _print, builtins.print = builtins.print, lambda *a, **k: None
    trace: dict = {"prompt": prompt, "config": cfg.to_dict(), "n_parameters": model.num_parameters()}

    # ------------------------------------------------------------------ 1 input
    say.header(1, "INPUT", "just a Python string")
    raw = prompt.encode("utf-8")
    print(f"   text   : {C['b']}{prompt!r}{C['x']}")
    print(f"   that is: {len(prompt)} characters = {len(raw)} bytes")
    print(f"   bytes  : {C['dim']}{' '.join(str(b) for b in raw[:16])} ...{C['x']}")
    trace["input"] = {"text": prompt, "n_chars": len(prompt), "bytes": list(raw)}

    # --------------------------------------------------------------- 2 tokenize
    say.header(2, "TOKENIZATION", "transparent_transformer/tokenizer.py")
    templated = format_prompt(prompt)
    ids = tok.encode(templated)
    pieces = [tok.token_str(i) for i in ids]
    print(f"   chat template wraps the text so the model knows who is speaking:\n   {C['dim']}{templated}{C['x']}\n")
    print("   " + " ".join(chip(p, i) for i, p in enumerate(pieces)))
    print(f"   {C['cyan']}{ids}{C['x']}")
    print(f"\n   {len(templated)} characters -> {len(ids)} tokens. Vocabulary size: {tok.vocab_size}")
    word = " weather"
    steps = tok.encode_steps(word)
    print(f"\n   how {word!r} gets merged, step by step:")
    for s in (steps if len(steps) <= 6 else steps[:3] + [["..."]] + steps[-2:]):
        print("      " + " ".join(chip(p, i) for i, p in enumerate(s)))
    trace["tokens"] = {"templated": templated, "ids": ids, "pieces": pieces,
                       "merge_demo": {"word": word, "steps": steps}, "vocab_size": tok.vocab_size}

    # ------------------------------------------------------------------ 3 embed
    say.header(3, "EMBEDDING", "transparent_transformer/embedding.py")
    x = np.array([ids])
    model.forward(x, capture=True)
    cap = model.captured
    tv, pv, s0 = cap["token_vectors"][0], cap["position_vectors"], cap["stream"][0][0]
    print(f"   each id looks up one row of a ({cfg.vocab_size} x {cfg.d_model}) table, then a position vector is added")
    print(f"   shape: ({len(ids)},) integers  ->  ({len(ids)}, {cfg.d_model}) floats\n")
    for i in range(min(4, len(ids))):
        nums = " ".join(f"{v:+.2f}" for v in s0[i, :8])
        print(f"   {chip(pieces[i], i):<28} id {ids[i]:>3} -> [{C['cyan']}{nums}{C['x']} ... ]")
    print(f"   {C['dim']}... {len(ids) - 4} more rows ...{C['x']}")
    trace["embedding"] = {"token": tv[:, :16].round(3).tolist(), "position": pv[:, :16].round(3).tolist(),
                          "sum": s0[:, :16].round(3).tolist(), "shape": [len(ids), cfg.d_model]}

    # ------------------------------------------------------------ 4 transformer
    say.header(4, "TRANSFORMER", "transparent_transformer/transformer.py")
    print(f"   {model.num_parameters():,} parameters in {cfg.n_layers} blocks. Each block ADDS to the stream:\n")
    norms = [np.linalg.norm(s[0], axis=-1) for s in cap["stream"]]
    print(f"   {'after':<14}{'size of the last token vector':<34}what was added")
    print(f"   {'embedding':<14}{norms[0][-1]:<34.2f}")
    for li in range(cfg.n_layers):
        a = np.linalg.norm(cap["attn_out"][li][0, -1])
        m = np.linalg.norm(cap["mlp_out"][li][0, -1])
        print(f"   {'block ' + str(li + 1):<14}{norms[li + 1][-1]:<34.2f}attention +{a:.2f}   MLP +{m:.2f}")
    # logit lens: read a prediction off the stream after EVERY layer, not just the last
    lens = []
    for st in cap["stream"]:
        h = model.ln_f.forward(st)[0, -1]
        pr = softmax((h @ model.embed.params["tok"].T).astype(np.float64))
        lens.append([{"piece": tok.token_str(j), "p": float(pr[j])} for j in np.argsort(-pr)[:5]])
    model.forward(x, capture=True)                     # restore the caches the lens overwrote
    trace["logit_lens"] = lens
    trace["last_vectors"] = [st[0, -1].round(3).tolist() for st in cap["stream"]]
    print("\n   logit lens: what the model would predict if it stopped after each layer")
    for name, row in zip(["embedding"] + [f"block {i+1}" for i in range(cfg.n_layers)], lens):
        print(f"   {name:<10} " + "  ".join(f"{c['piece']!r} {c['p']:.0%}" for c in row[:3]))
    trace["transformer"] = {
        "stream_norms": [n.round(3).tolist() for n in norms],
        "attn_norms": [np.linalg.norm(a[0], axis=-1).round(3).tolist() for a in cap["attn_out"]],
        "mlp_norms": [np.linalg.norm(m[0], axis=-1).round(3).tolist() for m in cap["mlp_out"]],
    }

    # -------------------------------------------------------------- 5 attention
    say.header(5, "CLOSE-UP: ATTENTION", "transparent_transformer/attention.py")
    print(f"   the LAST token {chip(pieces[-1], len(ids) - 1)} must predict the first word of the answer.")
    print("   which earlier tokens is it reading from?\n")
    for li in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            w = cap["attention"][li][0, h, -1]
            top = np.argsort(-w)[:3]
            desc = "  ".join(f"{chip(pieces[j], j)} {w[j]:.0%}" for j in top)
            print(f"   block {li + 1} head {h + 1}:  {desc}")
    trace["attention"] = [[a[0, h].round(3).tolist() for h in range(cfg.n_heads)] for a in cap["attention"]]

    # ------------------------------------------------- 6-8 how it got this way
    say.header(6, "PRETRAINING (done earlier: why the weights know things)", "transparent_transformer/pretrain.py")
    base_out = chat(base, tok, prompt)
    print("   the BASE model (pretraining only), given the same prompt, just keeps writing text:")
    print(f"   {C['coral']}{base_out!r}{C['x']}")
    say.header(7, "BACKPROPAGATION (how each weight was set)", "transparent_transformer/*.py -> backward()")
    print("   see stages/07_backpropagation/run.py for one full training step in slow motion")
    say.header(8, "ALIGNMENT (done earlier: why it answers, not rambles)", "transparent_transformer/alignment.py")
    print("   SFT taught the chat format. DPO taught it to prefer honest answers.")
    trace["base_output"] = base_out
    for name in ("pretrain_log", "sft_log", "dpo_log"):
        p = paths.ARTIFACTS / f"{name}.json"
        if p.exists():
            trace[name] = _thin(json.loads(p.read_text()))

    # --------------------------------------------------------------- 9 sampling
    say.header(9, "SAMPLING", "transparent_transformer/sampling.py")
    print(f"   temperature={temperature}, top_k=20, top_p=0.95. One full forward pass PER generated token.\n")
    rng = np.random.default_rng(seed)
    cur, gen = list(ids), []
    for n in range(40):
        logits = model.forward(np.array([cur[-cfg.context_length:]]))[0, -1]
        t, info = sample_next(logits, temperature, 20, 0.95, rng)
        order = np.argsort(-info["raw_probs"])[:5]
        cands = [{"id": int(j), "piece": tok.token_str(j), "p_raw": float(info["raw_probs"][j]),
                  "p_final": float(info["final_probs"][j])} for j in order]
        gen.append({"id": t, "piece": tok.token_str(t), "candidates": cands,
                    "logit_range": [float(logits.min()), float(logits.max())]})
        if n < 3:
            print(f"   step {n + 1}: the model's top guesses for the next token")
            for c in cands:
                mark = f"{C['mint']} <- chosen{C['x']}" if c["id"] == t else ""
                print(f"      {c['piece']!r:>12} {c['p_final']:>6.1%} {C['amber']}{bar(c['p_final'])}{C['x']}{mark}")
            print()
            say.pause()
        if t == tok.special["<|end|>"]:
            break
        cur.append(t)
    print(f"   {C['dim']}... and so on, one token at a time, until the model emits <|end|>{C['x']}")
    trace["generation"] = gen

    # The aligned model is very sure of itself, so the dice rarely matter. For contrast, look at the
    # FIRST answer token of the SFT checkpoint (before DPO): it is torn between two answer styles.
    demo = {}
    for label, path in (("sft", paths.SFT_MODEL), ("aligned", paths.ALIGNED_MODEL)):
        lg = GPT.load(path).forward(np.array([ids]))[0, -1]
        _, info = sample_next(lg, 1.0, None, None, np.random.default_rng(0))
        demo[label] = [{"piece": tok.token_str(j), "p": float(info["raw_probs"][j])}
                       for j in np.argsort(-info["raw_probs"])[:5]]
    trace["sampling_demo"] = demo
    print(f"\n   when do the dice matter? first answer token, temperature 1.0:")
    for label, title in (("sft", "before DPO (undecided: ' R' starts 'Right now it is 75 degrees...')"),
                         ("aligned", "after DPO  (decided:   ' I' starts 'I cannot see live weather data...')")):
        print(f"   {C['dim']}{title}{C['x']}")
        for c in demo[label][:2]:
            print(f"      {c['piece']!r:>12} {c['p']:>6.1%} {C['amber']}{bar(c['p'])}{C['x']}")

    # ----------------------------------------------------------------- 10 output
    say.header(10, "OUTPUT", "transparent_transformer/tokenizer.py -> decode()")
    out_ids = [g["id"] for g in gen if g["id"] != tok.special["<|end|>"]]
    answer = tok.decode(out_ids).strip()
    print("   ", end="")
    for i, g in enumerate(gen):
        if g["id"] != tok.special["<|end|>"]:
            print(chip(g["piece"], i), end=" ", flush=True)
            say.pause(0.08)
    print(f"\n\n   {C['b']}{C['mint']}{answer}{C['x']}\n")
    trace["output"] = answer

    if quiet:
        builtins.print = _print
    return trace


def _thin(log: dict, keep: int = 150) -> dict:
    """Down-sample long curves so the trace file stays small."""
    out = {}
    for k, v in log.items():
        if isinstance(v, list) and len(v) > keep and not isinstance(v[0], dict):
            idx = np.linspace(0, len(v) - 1, keep).astype(int)
            out[k] = [round(v[i], 4) if isinstance(v[i], float) else v[i] for i in idx]
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace one prompt through every stage of the model.")
    ap.add_argument("prompt", nargs="?", default="What is the weather in Los Angeles?")
    ap.add_argument("--fast", action="store_true", help="no animation")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    trace = run(a.prompt, animate=not a.fast, temperature=a.temperature, seed=a.seed)
    paths.DOCS.mkdir(exist_ok=True)
    (paths.ARTIFACTS / "trace.json").write_text(json.dumps(trace))
    name, var = ("trace_real.js", "TRACE_REAL") if paths.REAL else ("trace.js", "TRACE")
    (paths.DOCS / name).write_text(f"window.{var} = " + json.dumps(trace) + ";\n")
    print(f"{C['dim']}full trace saved -> docs/trace.js  (open docs/index.html to replay it visually){C['x']}")


if __name__ == "__main__":
    main()
