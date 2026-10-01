"""
STAGE 8g - REINFORCEMENT LEARNING AGAINST THE REWARD MODEL, WITH AND WITHOUT A LEASH
====================================================================================
Run:  python -m transparent_transformer.rlhf          (about eight minutes)

The last step of the InstructGPT recipe: the model writes answers, the reward model (stage 8f) scores them,
and reinforcement learning makes high-scoring answers more likely. Production systems use PPO; we use its
simplest ancestor, REINFORCE with a baseline, gradient written by hand:

    advantage = reward - average reward in the batch
    loss      = - advantage * log p(answer | prompt)          (push up above-average answers)

Stage 8f showed the reward model has blind spots. Optimise hard against it and the model will find them.
The standard defence is a KL penalty, a leash: every answer's reward is reduced by
    beta * (log p_model(answer) - log p_original(answer))
so drifting far from the original model costs reward. We train twice, without and with the leash, and
grade both with the constitution (stage 8d) on held-out prompts, which is what we actually care about.
"""
from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from . import paths
from .alignment import encode_example, format_prompt, pad_batch
from .loss import log_softmax, sequence_logprob
from .optimizer import AdamW, clip_gradients
from .reward_model import RewardModel, evaluate, features
from .sampling import generate
from .self_improve import measurement_wordings, practice_prompts, score
from .tokenizer import BPETokenizer
from .transformer import GPT

sys.path.insert(0, str(paths.ROOT / "data"))


def load_rm() -> RewardModel:
    d = json.loads((paths.ARTIFACTS / "reward_model.json").read_text())
    return RewardModel(np.array(d["w"]), d["b"], np.array(d["mu"]), np.array(d["sd"]))


def constitution_score(model, tok, prompts) -> dict:
    """Greedy answers to held-out practice-style prompts, graded by the six principles. Share of principles kept."""
    kept, total, clear = 0, 0, 0
    for kind, p, meta in prompts:
        a = tok.decode(generate(model, tok.encode(format_prompt(p)), 40, tok.special["<|end|>"], temperature=0.0)).strip()
        sc = score(kind, p, a, meta)
        kept += sum(v > 0 for v in sc.values()); total += len(sc); clear += sc.get("clear", 0) > 0
    return {"principles_kept": kept / total, "clear_sentences": clear / len(prompts)}


def train(beta: float, steps: int, rm: RewardModel, tok, prompts, heldout, lr=2e-5, batch=12) -> dict:
    model, ref = GPT.load(paths.ALIGNED_MODEL), GPT.load(paths.ALIGNED_MODEL)
    opt, rng, stop = AdamW(model.parameters(), lr=lr, weight_decay=0.0), random.Random(1), tok.special["<|end|>"]
    log = {"beta": beta, "reward": [], "kl": [], "samples": []}
    for step in range(steps):
        chosen = rng.sample(prompts, batch)
        seqs, answers = [], []
        for i, (_, p, _) in enumerate(chosen):
            new = generate(model, tok.encode(format_prompt(p)), 30, stop, temperature=1.0, top_k=None, top_p=None, seed=step * 100 + i)
            a = tok.decode(new).strip()
            answers.append(a)
            seqs.append(encode_example(tok, p, a))
        seqs = [(ids[-(model.cfg.context_length + 1):], m[-(model.cfg.context_length + 1):]) for ids, m in seqs]
        x, y, mask = pad_batch(seqs, stop)
        logits = model.forward(x)
        logp = sequence_logprob(logits, y, mask)
        ref_logp = sequence_logprob(ref.forward(x), y, mask)
        rm_scores = np.array([rm.score(features(ref, tok, p, a)) for (_, p, _), a in zip(chosen, answers)])
        kl = logp - ref_logp                                        # how far each answer has drifted, in log-probability
        reward = rm_scores - beta * kl
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)
        # d(-adv * logp)/dlogits = -adv * (one_hot(target) - probabilities), on answer tokens only
        dlogits = np.exp(log_softmax(logits))
        np.put_along_axis(dlogits, y[..., None], np.take_along_axis(dlogits, y[..., None], axis=-1) - 1.0, axis=-1)
        dlogits *= (mask * adv[:, None])[..., None] / batch
        model.backward(dlogits.astype(np.float32))
        grads = model.gradients()
        clip_gradients(grads, 1.0)
        opt.step(grads, lr=lr)
        log["reward"].append(float(rm_scores.mean())); log["kl"].append(float(kl.mean()))
        if step % 25 == 0 or step == steps - 1:
            log["samples"].append({"step": step, "prompt": chosen[0][1], "answer": answers[0], "reward": round(float(rm_scores[0]), 2)})
            print(f"   beta {beta:<4} step {step:>3} | reward model says {rm_scores.mean():+6.2f} | drift (KL) {kl.mean():+6.2f} | e.g. {answers[0][:60]!r}")
    log["after"] = {**constitution_score(model, tok, heldout), **{k: float(v) for k, v in evaluate(model, tok, rm, "greedy").items()}}
    return log


def main() -> None:
    t0 = time.time()
    tok = BPETokenizer.load(paths.TOKENIZER)
    rm = load_rm()
    test = measurement_wordings()
    pool = [p for p in practice_prompts(random.Random(0)) if p[1] not in test]
    heldout = [p for p in practice_prompts(random.Random(42)) if p[1] not in test and p not in pool][:60]
    base = GPT.load(paths.ALIGNED_MODEL)
    before = {**constitution_score(base, tok, heldout), **{k: float(v) for k, v in evaluate(base, tok, rm, "greedy").items()}}
    print(f"before: {json.dumps({k: round(v, 2) for k, v in before.items()})}\n")
    runs = {}
    for beta in (0.0, 0.5):
        print(f"training with KL penalty beta = {beta} ({'no leash' if beta == 0 else 'leash on'})")
        runs[str(beta)] = train(beta, 120, rm, tok, pool, heldout)
        print(f"   after: {json.dumps({k: round(v, 2) for k, v in runs[str(beta)]['after'].items()})}\n")
    out = {"before": before, "runs": runs}
    (paths.ARTIFACTS / "rlhf_log.json").write_text(json.dumps(out, indent=1))
    print(f"saved artifacts/rlhf_log.json ({time.time() - t0:.0f}s). Neither model replaces the aligned one: this stage is an experiment.")


if __name__ == "__main__":
    main()
