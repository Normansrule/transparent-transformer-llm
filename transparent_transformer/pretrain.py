"""
STAGE 6 - PRETRAINING: LEARNING BY PREDICTING THE NEXT TOKEN
============================================================
Run:  python -m transparent_transformer.pretrain

The entire training signal is one game played millions of times:

    "Here are some tokens from a real document. Guess the next one."

    input  : [ The] [ weather] [ in] [ Los] [ Angeles] [ is] [ usually]
    target : [ weather] [ in] [ Los] [ Angeles] [ is] [ usually] [ sunny]
                                   (the same text, shifted left by one)

No labels, no humans, no questions and answers. Yet to get good at this game
the model is forced to absorb spelling, grammar, and facts ("Los Angeles" ->
"sunny"). The result is called a BASE MODEL: it continues text, it does not
follow instructions. Stage 8 fixes that.

One training step =
    1. forward   run a batch through the model               (stage 4)
    2. loss      measure how surprised it was                (loss.py)
    3. backward  work out each weight's share of the blame   (stage 7)
    4. update    nudge every weight the other way            (optimizer.py)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import paths
from .config import PRESETS, Config
from .loss import cross_entropy
from .optimizer import AdamW, clip_gradients, cosine_schedule
from .sampling import generate
from .tokenizer import BPETokenizer
from .transformer import GPT

PROBE = "The weather in Los Angeles is"


def get_batch(data: np.ndarray, batch_size: int, T: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - T - 1, size=batch_size)
    x = np.stack([data[s:s + T] for s in starts])
    y = np.stack([data[s + 1:s + T + 1] for s in starts])    # targets = inputs shifted by one
    return x, y


def train_tokenizer(vocab_size: int, verbose: bool = True) -> BPETokenizer:
    text = (paths.DATA / "pretrain.txt").read_text()
    tok = BPETokenizer()
    if verbose:
        print(f"Training Byte Pair Encoding (BPE) tokenizer: 256 bytes -> {vocab_size} tokens")
    log = tok.train(text, vocab_size, verbose=verbose)
    paths.ARTIFACTS.mkdir(exist_ok=True)
    tok.save(paths.TOKENIZER)
    (paths.ARTIFACTS / "tokenizer_log.json").write_text(json.dumps(log, indent=1))
    return tok


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6: pretrain the base model on raw text.")
    ap.add_argument("--preset", choices=list(PRESETS), default=os.environ.get("TT_PRESET") or ("small" if paths.REAL else "tiny"))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tokenizer-only", action="store_true", help="stage 2 only: learn the BPE merges and stop")
    args = ap.parse_args()

    settings, steps, batch, _, _ = PRESETS[args.preset]
    args.steps, args.batch_size = args.steps or steps, args.batch_size or batch
    cfg = Config(**settings)
    print(f"track: {paths.MODEL}   preset: {args.preset}   data: {paths.DATA.name}/   weights: {paths.ARTIFACTS.name}/")
    if args.tokenizer_only:
        train_tokenizer(cfg.vocab_size)
        return
    tok = BPETokenizer.load(paths.TOKENIZER) if paths.TOKENIZER.exists() else train_tokenizer(cfg.vocab_size)
    text = (paths.DATA / "pretrain.txt").read_text()
    data = np.array(tok.encode(text), dtype=np.int64)
    split = int(len(data) * 0.95)
    train, val = data[:split], data[split:]
    print(f"corpus: {len(text):,} characters -> {len(data):,} tokens "
          f"({len(text)/len(data):.2f} characters per token)")

    model = GPT(cfg)
    print(f"model : {model.num_parameters():,} parameters  "
          f"({cfg.n_layers} blocks, d_model={cfg.d_model}, {cfg.n_heads} heads)")
    print(f"a model that guesses uniformly at random scores loss = ln({cfg.vocab_size}) = {np.log(cfg.vocab_size):.2f}\n")

    opt = AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(cfg.seed)
    log = {"steps": [], "train_loss": [], "val_steps": [], "val_loss": [], "samples": [], "grad_norm": []}
    snapshots = {0, 20, 100, 300, args.steps}
    every = 100 if args.steps <= 2000 else 250
    t0 = time.time()

    for step in range(args.steps + 1):
        if step in snapshots:                              # what does the model write right now?
            out = generate(model, tok.encode(PROBE), max_new_tokens=14, temperature=0.7, seed=1)
            log["samples"].append({"step": step, "text": PROBE + tok.decode(out)})
            print(f"   [step {step}] model writes: {PROBE}\033[36m{tok.decode(out)!s}\033[0m".replace("\n", "\\n"))
        if step % 100 == 0:
            vx, vy = get_batch(val, 32, cfg.context_length, np.random.default_rng(0))
            vloss, _ = cross_entropy(model.forward(vx), vy)
            log["val_steps"].append(step)
            log["val_loss"].append(vloss)
        if step == args.steps:
            break

        x, y = get_batch(train, args.batch_size, cfg.context_length, rng)
        logits = model.forward(x)                          # 1. forward
        loss, dlogits = cross_entropy(logits, y)           # 2. loss
        model.backward(dlogits)                            # 3. backward
        grads = model.gradients()
        gnorm = clip_gradients(grads, 1.0)
        opt.step(grads, lr=cosine_schedule(step, args.steps, args.lr))   # 4. update

        log["steps"].append(step)
        log["train_loss"].append(loss)
        log["grad_norm"].append(gnorm)
        if step % every == 0:
            print(f"step {step:>5} | train loss {loss:.3f} | val loss {vloss:.3f} | {time.time()-t0:5.1f}s")

    model.save(paths.BASE_MODEL)
    (paths.ARTIFACTS / "pretrain_log.json").write_text(json.dumps(log))
    print(f"\nsaved base model -> {paths.BASE_MODEL.relative_to(paths.ROOT)}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
