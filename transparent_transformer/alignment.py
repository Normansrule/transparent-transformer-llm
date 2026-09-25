r"""
STAGE 8 - ALIGNMENT: FROM "TEXT CONTINUER" TO "ASSISTANT"
=========================================================
Run:  python -m transparent_transformer.alignment sft
      python -m transparent_transformer.alignment dpo

The base model from stage 6 knows a lot about weather but has terrible
manners. Ask it a question and it does not answer: it carries on writing
weather documents, because continuing documents is all it has ever done. Alignment changes the
model's behaviour without (much) changing its knowledge. Two steps:

8a. Supervised Fine-Tuning (SFT)
    Same next-token game as pretraining, new data: conversations written in a
    fixed chat template

        <|user|>What is the weather in Los Angeles?<|assistant|> ... <|end|>
        \_______________ prompt: loss IGNORED ______/\__ response: loss COUNTED __/

    The loss mask is the only new idea. We do not want to teach the model to
    write user questions, only to write good answers.

8b. Direct Preference Optimization (DPO)
    Our SFT data is imperfect on purpose: half of its answers pretend to know
    the live temperature. DPO shows the model PAIRS of answers to the same
    prompt - one chosen, one rejected - and adjusts the weights so the chosen
    one becomes relatively more likely than it was before:

        z    = beta * [ (logp(chosen) - logp(rejected))            <- this model
                      - (logp_ref(chosen) - logp_ref(rejected)) ]  <- frozen copy of the SFT model
        loss = -log(sigmoid(z))

    The frozen reference keeps the model from drifting too far from what it
    already knows. This is the same recipe used (at vastly larger scale)
    to make production assistants more helpful and honest. It is a simpler
    cousin of Reinforcement Learning from Human Feedback (RLHF).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import paths
from .loss import cross_entropy, log_softmax, sequence_logprob
from .optimizer import AdamW, clip_gradients, cosine_schedule
from .sampling import generate
from .tokenizer import BPETokenizer
from .transformer import GPT

DEMO_PROMPT = "What is the weather in Los Angeles?"


# ------------------------------------------------------------------ chat template
def format_prompt(prompt: str) -> str:
    return f"<|user|>{prompt}<|assistant|>"


def format_example(prompt: str, response: str) -> str:
    return f"{format_prompt(prompt)} {response}<|end|>"


def encode_example(tok: BPETokenizer, prompt: str, response: str):
    """Returns (ids, mask) where mask is 1 on response tokens - the only ones that are trained on."""
    p = tok.encode(format_prompt(prompt))
    r = tok.encode(f" {response}<|end|>")
    return p + r, [0] * len(p) + [1] * len(r)


def pad_batch(examples, pad_id: int):
    """Stack variable-length sequences into rectangles. Padding is masked out so it never affects the loss."""
    T = max(len(ids) for ids, _ in examples) - 1
    x = np.full((len(examples), T), pad_id, dtype=np.int64)
    y = np.full((len(examples), T), pad_id, dtype=np.int64)
    m = np.zeros((len(examples), T), dtype=np.float32)
    for i, (ids, mask) in enumerate(examples):
        n = len(ids) - 1
        x[i, :n], y[i, :n], m[i, :n] = ids[:-1], ids[1:], mask[1:]   # predict token t+1 from tokens <= t
    return x, y, m


def chat(model: GPT, tok: BPETokenizer, prompt: str, **kw) -> str:
    kw.setdefault("temperature", 0.0)
    out = generate(model, tok.encode(format_prompt(prompt)), max_new_tokens=64,
                   stop_id=tok.special["<|end|>"], **kw)
    return tok.decode(out).strip()


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (paths.DATA / name).read_text().splitlines() if line.strip()]


def honesty_rate(model: GPT, tok: BPETokenizer, n: int = 30) -> float:
    """Fraction of sampled weather answers that admit they cannot see live data."""
    prompts = [r["prompt"] for r in load_jsonl("prefs.jsonl")[:n]]
    hits = sum(chat(model, tok, p, temperature=1.0, top_k=None, top_p=None, seed=i).startswith("I cannot")
               for i, p in enumerate(prompts))
    return hits / len(prompts)


# ------------------------------------------------------------------------ 8a. SFT
def run_sft(steps: int, batch_size: int, lr: float) -> None:
    tok = BPETokenizer.load(paths.TOKENIZER)
    model = GPT.load(paths.BASE_MODEL)
    rows = load_jsonl("sft.jsonl")
    examples = [encode_example(tok, r["prompt"], r["response"]) for r in rows]
    pad = tok.special["<|end|>"]
    print(f"Supervised Fine-Tuning (SFT) on {len(rows)} conversations\n")
    print(f"BEFORE  base model, asked {DEMO_PROMPT!r}:\n        \033[33m{chat(model, tok, DEMO_PROMPT)!r}\033[0m\n")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    log = {"steps": [], "loss": [], "before": chat(model, tok, DEMO_PROMPT)}
    t0 = time.time()
    for step in range(steps):
        batch = [examples[i] for i in rng.integers(0, len(examples), size=batch_size)]
        x, y, mask = pad_batch(batch, pad)
        loss, dlogits = cross_entropy(model.forward(x), y, mask)     # <- mask: only the response counts
        model.backward(dlogits)
        grads = model.gradients()
        clip_gradients(grads, 1.0)
        opt.step(grads, lr=cosine_schedule(step, steps, lr, warmup=20))
        log["steps"].append(step)
        log["loss"].append(loss)
        if step % (50 if steps <= 400 else 250) == 0 or step == steps - 1:
            print(f"step {step:>4} | loss on response tokens {loss:.3f} | {time.time()-t0:4.1f}s")

    log["after"] = chat(model, tok, DEMO_PROMPT)
    log["honesty_rate"] = honesty_rate(model, tok)
    print(f"\nAFTER   SFT model:\n        \033[36m{log['after']!r}\033[0m")
    print(f"        answers that admit 'I cannot see live weather data': {log['honesty_rate']:.0%}")
    model.save(paths.SFT_MODEL)
    (paths.ARTIFACTS / "sft_log.json").write_text(json.dumps(log))
    print(f"\nsaved -> {paths.SFT_MODEL.relative_to(paths.ROOT)}")


# ------------------------------------------------------------------------ 8b. DPO
def run_dpo(steps: int, batch_size: int, lr: float, beta: float) -> None:
    tok = BPETokenizer.load(paths.TOKENIZER)
    model = GPT.load(paths.SFT_MODEL)        # the "policy": this one learns
    reference = model.copy()                 # frozen snapshot: the anchor
    rows = load_jsonl("prefs.jsonl")
    pad = tok.special["<|end|>"]
    pairs = [(encode_example(tok, r["prompt"], r["chosen"]), encode_example(tok, r["prompt"], r["rejected"]))
             for r in rows]
    before_rate = honesty_rate(model, tok)
    print(f"Direct Preference Optimization (DPO) on {len(rows)} chosen/rejected pairs, beta={beta}")
    print(f"honest answers before DPO: {before_rate:.0%}\n")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    log = {"steps": [], "loss": [], "accuracy": [], "margin": [], "honesty_before": before_rate}
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, len(pairs), size=batch_size)
        # put chosen and rejected answers in ONE batch: rows [0:B] chosen, rows [B:2B] rejected
        x, y, mask = pad_batch([pairs[i][0] for i in idx] + [pairs[i][1] for i in idx], pad)
        B = batch_size

        logits = model.forward(x)
        logp = sequence_logprob(logits, y, mask)                     # log p(answer | prompt), this model
        ref_logp = sequence_logprob(reference.forward(x), y, mask)   # same, frozen reference

        z = beta * ((logp[:B] - logp[B:]) - (ref_logp[:B] - ref_logp[B:]))
        sig = 1.0 / (1.0 + np.exp(-z))
        loss = float(-np.log(sig + 1e-12).mean())

        # ---- gradient, by hand -------------------------------------------------------------
        # dloss/dz = -(1 - sig);  dz/dlogp_chosen = +beta;  dz/dlogp_rejected = -beta
        # dlogp/dlogits at each response position = one_hot(target) - probabilities
        coef = np.concatenate([-(1 - sig) * beta, (1 - sig) * beta]) / B          # (2B,)
        dlogits = -np.exp(log_softmax(logits))                                     # -probabilities
        np.put_along_axis(dlogits, y[..., None],
                          np.take_along_axis(dlogits, y[..., None], axis=-1) + 1.0, axis=-1)   # + one_hot
        dlogits *= (mask * coef[:, None])[..., None]
        model.backward(dlogits.astype(np.float32))
        grads = model.gradients()
        clip_gradients(grads, 1.0)
        opt.step(grads, lr=cosine_schedule(step, steps, lr, warmup=10))

        log["steps"].append(step)
        log["loss"].append(loss)
        log["accuracy"].append(float((z > 0).mean()))
        log["margin"].append(float(z.mean() / beta))
        if step % 20 == 0 or step == steps - 1:
            print(f"step {step:>4} | loss {loss:.3f} | prefers chosen in {(z > 0).mean():4.0%} of pairs "
                  f"| margin {z.mean()/beta:+6.2f} | {time.time()-t0:4.1f}s")

    log["honesty_after"] = honesty_rate(model, tok)
    log["after"] = chat(model, tok, DEMO_PROMPT)
    print(f"\nhonest answers after DPO : {log['honesty_after']:.0%}   (was {before_rate:.0%})")
    print(f"aligned model, asked {DEMO_PROMPT!r}:\n        \033[32m{log['after']!r}\033[0m")
    model.save(paths.ALIGNED_MODEL)
    (paths.ARTIFACTS / "dpo_log.json").write_text(json.dumps(log))
    print(f"\nsaved -> {paths.ALIGNED_MODEL.relative_to(paths.ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8: align the base model.")
    ap.add_argument("method", choices=["sft", "dpo"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--beta", type=float, default=0.2)
    a = ap.parse_args()
    sft_steps, dpo_steps = (1500, 150) if paths.REAL else (300, 100)     # the scraped dataset is about 30x larger
    if a.method == "sft":
        run_sft(a.steps or sft_steps, a.batch_size, a.lr or 1e-3)
    else:
        run_dpo(a.steps or dpo_steps, a.batch_size, a.lr or 3e-5, a.beta)


if __name__ == "__main__":
    main()
