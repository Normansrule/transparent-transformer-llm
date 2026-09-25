"""
The "ask the model" bot. A GitHub Actions workflow calls this when someone opens an issue whose title
starts with "Ask:". It runs the prompt through every stage and prints a Markdown reply.
The issue title arrives ONLY through the TITLE environment variable (never pasted into a shell command).
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from transparent_transformer.trace import run  # noqa: E402

title = os.environ.get("TITLE", "Ask: What is the weather in Los Angeles?")
prompt = re.sub(r"^\s*ask\s*:\s*", "", title, flags=re.I)
prompt = "".join(ch for ch in prompt if ch.isprintable()).strip()[:120] or "What is the weather in Los Angeles?"
t = run(prompt, animate=False, quiet=True)


def cell(s: str) -> str:
    return "`" + s.replace("`", "'").replace("|", "¦").replace(" ", "·").replace("\n", "↵") + "`"


att = np.array(t["attention"])                     # (layers, heads, T, T)
L, H = att.shape[:2]
heads = []
for l in range(L):
    for h in range(H):
        row = att[l, h, -1]
        j = int(row.argmax())
        heads.append(f"| block {l+1}, head {h+1} | {cell(t['tokens']['pieces'][j])} | {row[j]:.0%} |")
first = t["generation"][0]["candidates"]
safe = prompt.replace("`", "'")

print(f"""### The model says

> **{t['output']}**

<details open><summary><b>1-2. Your text became {len(t['tokens']['ids'])} tokens</b></summary>

| {' | '.join(cell(p) for p in t['tokens']['pieces'])} |
|{'---|' * len(t['tokens']['ids'])}
| {' | '.join(str(i) for i in t['tokens']['ids'])} |

</details>

<details><summary><b>3-5. Inside the transformer: what the last token looked at</b></summary>

Each token became a vector of {t['config']['d_model']} numbers and flowed through {L} blocks ({t['n_parameters']:,} weights).
Just before answering, each attention head was reading mostly from:

| head | looked hardest at | weight |
|---|---|---|
{chr(10).join(heads)}

</details>

<details><summary><b>6-8. The same prompt through the three checkpoints</b></summary>

| checkpoint | answer |
|---|---|
| base (pretraining only) | {cell(t['base_output'][:110])} |
| after Supervised Fine-Tuning (SFT), first-token odds | {', '.join(f"{cell(c['piece'])} {c['p']:.0%}" for c in t['sampling_demo']['sft'][:2])} |
| after Direct Preference Optimization (DPO), first-token odds | {', '.join(f"{cell(c['piece'])} {c['p']:.0%}" for c in t['sampling_demo']['aligned'][:2])} |

</details>

<details><summary><b>9-10. Sampling: the top candidates for the first word</b></summary>

| candidate | probability |
|---|---|
{chr(10).join(f"| {cell(c['piece'])} | {c['p_final']:.1%} |" for c in first)}

Then the model ran {len(t['generation'])} times in total, adding one token per run, until it produced the end token.

</details>

Prompt received: `{safe}`. This model has {t['n_parameters']:,} weights and only knows weather small talk, so expect confident nonsense about anything else. That is part of the lesson: see [stage 8](../blob/main/stages/08_alignment/).
""")
