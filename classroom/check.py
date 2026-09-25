"""
The auto-grader.   python classroom/check.py              grade classroom/exercises/
                   python classroom/check.py --solutions  grade the reference solutions (should be 10/10)

On GitHub, the "homework" workflow runs this on every push and shows the table on the run's Summary page.
Exit code is 1 only if an exercise you ATTEMPTED is wrong; exercises not started yet never fail the build.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
folder = HERE / ("solutions" if "--solutions" in sys.argv else "exercises")
rng = np.random.default_rng(0)


def load(name):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t01(m): return m.to_bytes("Hi") == [72, 105] and m.to_bytes("é") == [195, 169]
def t02(m): return m.merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9] and m.merge([1, 1, 1], (1, 1), 5) == [5, 1]
def t03(m):
    tok, pos = rng.normal(size=(10, 4)), rng.normal(size=(6, 4))
    return np.allclose(m.embed([3, 3, 7], tok, pos), tok[[3, 3, 7]] + pos[:3])
def t04(m):
    x = rng.normal(size=(3, 8)) * 5 + 2
    y = m.layer_norm(x, np.ones(8), np.zeros(8))
    return np.allclose(y.mean(-1), 0, atol=1e-6) and np.allclose(y.std(-1), 1, atol=1e-3) and np.allclose(m.layer_norm(x, 2 * np.ones(8), np.ones(8)), 2 * y + 1)
def t05(m):
    w = m.causal_attention_weights(rng.normal(size=(5, 5)))
    return np.allclose(w.sum(-1), 1) and np.allclose(np.triu(w, 1), 0) and w[0, 0] == 1 and (w >= 0).all()
def t06(m):
    p = np.array([[0.5, 0.5], [0.25, 0.75]])
    return abs(m.next_token_loss(p, np.array([0, 1])) - (-(np.log(0.5) + np.log(0.75)) / 2)) < 1e-9
def t07(m):
    x, W, dy = rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=(4, 5))
    dx, dW = m.linear_backward(x, W, dy)
    return np.allclose(dx, dy @ W.T) and np.allclose(dW, x.T @ dy)
def t08(m):
    return abs(m.dpo_loss(-5, -5, -5, -5) - np.log(2)) < 1e-9 and m.dpo_loss(-2, -9, -5, -5) < m.dpo_loss(-9, -2, -5, -5)
def t09(m):
    p = m.temperature_top_k(np.array([1.0, 2.0, 3.0, 0.0]), 0.5, 2)
    e = np.exp(np.array([4.0, 6.0]) - 6.0); e /= e.sum()
    return np.allclose(p, [0, e[0], e[1], 0]) and abs(p.sum() - 1) < 1e-9
def t10(m):
    script = iter([4, 5, 6, 99, 7])
    seen = []
    out = m.generate(lambda ids: (seen.append(list(ids)), next(script))[1], [1, 2], end_id=99)
    return out == [4, 5, 6] and seen[1] == [1, 2, 4] and m.generate(lambda ids: 3, [0], end_id=99, max_new=4) == [3, 3, 3, 3]


LESSONS = ["Input", "Tokenization", "Embedding", "Transformer", "Attention", "Pretraining", "Backpropagation", "Alignment", "Sampling", "Output"]
rows, failed, done = [], False, 0
for i, lesson in enumerate(LESSONS, start=1):
    name = f"ex{i:02d}"
    try:
        ok = bool(globals()[f"t{i:02d}"](load(name)))
        status = "✅ passed" if ok else "❌ not right yet"
        failed |= not ok
        done += ok
    except NotImplementedError:
        status = "⬜ not started"
    except Exception as e:  # noqa: BLE001
        status, failed = f"❌ error: {type(e).__name__}: {e}"[:70], True
    rows.append((name, lesson, status))

table = "| exercise | lesson | result |\n|---|---|---|\n" + "\n".join(f"| `{n}.py` | {l} | {s} |" for n, l, s in rows)
summary = f"## Homework: {done} of 10 passed\n\n{table}\n"
print(summary)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(summary)
sys.exit(1 if failed else 0)
