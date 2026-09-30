"""
STAGE 8e - DISTILLATION: TEACH THE WEIGHTS WHAT THE HARNESS KNOWS
===================================================================
Run:  python -m transparent_transformer.distill          (about four minutes)

The bare model answers 0 of 12 messy questions ("whats LA like in summer", "how is summer in seatle?").
With the harness's normalizer in front of it, the same weights answer 9 of 12. Can the weights learn to do
it alone? This is DISTILLATION: a stronger system (the teacher: normalizer + model) produces answers,
we keep only the ones we can verify, and the plain model (the student) is fine-tuned to produce them
from the ORIGINAL messy question. Anthropic's early assistant research used a version called context
distillation, training a model to behave as if a long prompt were present when it is not.

Two traps this file guards against:
  * catastrophic forgetting: fine-tuning only on new data can erase old skills, so every batch mixes in
    the original conversations (replay);
  * fooling ourselves: three question templates are held out COMPLETELY, so the test asks questions
    shaped in ways the student never practised.
"""
from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from . import paths
from .alignment import chat, dpo_steps, encode_example, honesty_rate, load_jsonl, pad_batch, safety_rates
from .harness import Harness
from .loss import cross_entropy
from .optimizer import AdamW, clip_gradients, cosine_schedule
from .tokenizer import BPETokenizer
from .transformer import GPT

sys.path.insert(0, str(paths.ROOT / "data"))
from make_corpus import CITIES, HELD_OUT  # noqa: E402

TRAIN_TEMPLATES = ["{c} in {s}?", "hows {c} in {s}", "tell me about {s} in {c}", "{c} {s} weather", "what's {s} like in {c}",
                   "{c} during {s}?", "whats {c} like in {s}", "describe {s} in {c}", "{s} in {c}?", "how is {s} in {c}?"]
HELDOUT_TEMPLATES = ["any idea what {s} is like in {c}?", "{s} in {c}, what should i expect", "going to {c} in {s}. what's it like"]
ALIASES = {"Los Angeles": ["LA", "L.A."], "New York": ["NYC"], "San Francisco": ["SF"], "Mexico City": ["CDMX"], "Buenos Aires": ["BA"]}


def spellings(city: str, rng: random.Random) -> list[str]:
    """How people actually type a city: exact, lowercase, a nickname, or one typing mistake."""
    out = [city, city.lower()] + ALIASES.get(city, [])
    if len(city) >= 6:
        i = rng.randrange(1, len(city) - 1)
        out.append(city[:i] + city[i + 1:])                                     # a dropped letter
        j = rng.randrange(1, len(city) - 2)
        out.append(city[:j] + city[j + 1] + city[j] + city[j + 2:])             # two letters swapped
    return out


def messy_questions(templates, rng, per_city=6):
    known = [c for c in CITIES if c[0] not in HELD_OUT]
    out = []
    for c, _, _, summer, winter, *_ in known:
        for _ in range(per_city):
            season = rng.choice(["summer", "winter"])
            q = rng.choice(templates).format(c=rng.choice(spellings(c, rng)), s=season)
            out.append((q, c, summer if season == "summer" else winter))
    return out


def correct(answer: str, city: str, fact: str) -> bool:
    return city in answer and fact in answer


def accuracy(model, tok, questions) -> float:
    return sum(correct(chat(model, tok, q), c, f) for q, c, f in questions) / len(questions)


def measure(model, tok, fresh) -> dict:
    from .agent_eval import CASES, passed
    h = Harness(offline=True, tricks=())
    h.model = model
    messy = [(q, s) for cat, q, s in CASES if cat == "messy climate question"]
    clean = [(q, s) for cat, q, s in CASES if cat == "clean climate question"]
    sr = safety_rates(model, tok)
    def ask(q):
        h.memory = []                      # every test question starts a fresh conversation
        return h.reply(q)
    return {"messy_test_12": sum(passed(ask(q), s) for q, s in messy) / len(messy),
            "heldout_templates": accuracy(model, tok, fresh),
            "clean_climate": sum(passed(ask(q), s) for q, s in clean) / len(clean),
            "refuses_harmful": sr["refuses_harmful"], "refuses_safe": sr["refuses_safe"], "honest_sampled": honesty_rate(model, tok)}


def main() -> None:
    from .agent_eval import CASES
    rng = random.Random(0)
    tok = BPETokenizer.load(paths.TOKENIZER)
    student = GPT.load(paths.ALIGNED_MODEL)
    teacher = Harness(offline=True)                                   # every trick on: the normalizer does the work
    fresh = messy_questions(HELDOUT_TEMPLATES, random.Random(99), per_city=2)
    before = measure(student, tok, fresh)
    print(f"before: {json.dumps({k: round(v, 2) for k, v in before.items()})}\n")

    t0, test = time.time(), {q for _, q, _ in CASES}
    candidates = [x for x in messy_questions(TRAIN_TEMPLATES, rng) if x[0] not in test]
    rows, rejected = [], 0
    for q, city, fact in candidates:
        teacher.memory = []
        ans = teacher.reply(q)
        if correct(ans, city, fact):
            rows.append({"prompt": q, "response": ans})
        else:
            rejected += 1
    print(f"teacher (normalizer + model) answered {len(candidates)} messy questions: kept {len(rows)} verified answers, "
          f"threw away {rejected} ({time.time() - t0:.0f}s)")
    print(f"   e.g.  {rows[0]['prompt']!r}  ->  {rows[0]['response']!r}")

    replay = load_jsonl("sft.jsonl")
    new = [encode_example(tok, r["prompt"], r["response"]) for r in rows]
    old = [encode_example(tok, r["prompt"], r["response"]) for r in replay]
    limit = student.cfg.context_length + 1
    new, old = [e for e in new if len(e[0]) <= limit], [e for e in old if len(e[0]) <= limit]
    pad, steps, bs, lr = tok.special["<|end|>"], 300, 16, 5e-5
    opt, nrng, log = AdamW(student.parameters(), lr=lr, weight_decay=0.0), np.random.default_rng(0), {"loss": []}
    print(f"\nfine-tuning: {steps} steps, each batch half new (distilled) and half replayed old conversations")
    for step in range(steps):
        batch = [new[i] for i in nrng.integers(0, len(new), bs // 2)] + [old[i] for i in nrng.integers(0, len(old), bs // 2)]
        x, y, mask = pad_batch(batch, pad)
        loss, dlogits = cross_entropy(student.forward(x), y, mask)
        student.backward(dlogits)
        grads = student.gradients()
        clip_gradients(grads, 1.0)
        opt.step(grads, lr=cosine_schedule(step, steps, lr, warmup=20))
        log["loss"].append(loss)
    mid = measure(student, tok, fresh)
    print(f"after distilling:   {json.dumps({k: round(v, 2) for k, v in mid.items()})}")
    # Fine-tuning washed out what preference training had sharpened (our first run: honesty 30% -> 7%).
    # So, as in production recipes, preference training comes AFTER fine-tuning: re-run the original DPO pairs.
    prefs = load_jsonl("prefs.jsonl")
    pairs = [(encode_example(tok, r["prompt"], r["chosen"]), encode_example(tok, r["prompt"], r["rejected"])) for r in prefs]
    pairs = [q for q in pairs if max(len(q[0][0]), len(q[1][0])) <= limit]
    dpo_steps(student, student.copy(), pairs, pad, 150, 16, 3e-5, 0.1, verbose=False)
    after = measure(student, tok, fresh)
    print(f"after re-aligning:  {json.dumps({k: round(v, 2) for k, v in after.items()})}")
    log["after_distill_only"] = mid
    kept = (after["heldout_templates"] > before["heldout_templates"] and after["refuses_harmful"] >= before["refuses_harmful"]
            and after["refuses_safe"] <= before["refuses_safe"] and after["clean_climate"] >= before["clean_climate"]
            and after["honest_sampled"] >= before["honest_sampled"] - 0.05)
    examples = [(q, chat(student, tok, q)) for q, _, _ in fresh[:6]]
    log.update({"before": before, "after": after, "kept": kept, "distilled_rows": len(rows), "rejected": rejected,
                "heldout_templates": HELDOUT_TEMPLATES, "examples": examples})
    (paths.ARTIFACTS / "distill_log.json").write_text(json.dumps(log, indent=1))
    for q, a in examples[:4]:
        print(f"   {q!r:45} -> {a}")
    if kept:
        student.save(paths.ALIGNED_MODEL)
        print(f"\nkept: saved -> {paths.ALIGNED_MODEL.relative_to(paths.ROOT)}")
    else:
        print("\nNOT kept: a guard metric got worse or the held-out templates did not improve.")


if __name__ == "__main__":
    main()
