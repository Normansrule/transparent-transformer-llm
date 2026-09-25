"""
Keeps the stage pages honest: runs every stages/*/run.py and pastes its REAL output into the page,
and rebuilds the navigation header. Run after retraining:  python tools/refresh_docs.py
"""
import re
import subprocess
import sys
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parent.parent
USER, REPO = "Normansrule", "transparent-transformer-llm"
QUIZ = json.loads((ROOT / "classroom" / "quiz.json").read_text())


def badge(label, msg, colour, url):
    img = f"https://img.shields.io/badge/{label}-{msg}-{colour}?style=for-the-badge".replace(" ", "%20")
    return f'<a href="{url}"><img src="{img}" alt="{label}: {msg}"></a>'

ANSI = re.compile(r"\x1b\[[0-9;]*m")
dirs = sorted(p for p in (ROOT / "stages").iterdir() if p.is_dir())


def pretty(d: Path) -> str:
    return d.name.split("_", 1)[1].replace("_", " ")


for i, d in enumerate(dirs):
    readme = d / "README.md"
    text = readme.read_text()

    back = f'<a href="../{dirs[i-1].name}/">&larr; {pretty(dirs[i-1])}</a>' if i else '<a href="../../README.md">&larr; overview</a>'
    fwd = f'<a href="../{dirs[i+1].name}/"><b>next: {pretty(dirs[i+1])} &rarr;</b></a>' if i < len(dirs) - 1 \
        else '<a href="../../README.md"><b>finish: overview &rarr;</b></a>'
    nav = (f'<!-- NAV -->\n<img src="../../assets/pipeline_{i+1:02d}.svg" width="100%" '
           f'alt="Map of the ten stages. You are at stage {i+1}: {pretty(d)}. A carriage carries the data in from the previous stage.">\n\n'
           f'<p align="center">{back} &nbsp;&nbsp;&middot;&nbsp;&nbsp; stage {i+1} of {len(dirs)} &nbsp;&nbsp;&middot;&nbsp;&nbsp; {fwd}</p>\n<!-- /NAV -->')
    n = i + 1
    do = ("<!-- DO -->\n<p align=\"center\">\n"
          + badge("🧪 try it live", "in your browser", "FFB238", f"https://{USER}.github.io/{REPO}/#{n}") + "\n"
          + badge("📓 see the code run", "notebook", "5CC8FF", f"../../notebooks/{d.name}.ipynb") + "\n"
          + badge("▶ run it yourself", "Colab", "C9A7FF", f"https://colab.research.google.com/github/{USER}/{REPO}/blob/main/notebooks/{d.name}.ipynb") + "\n"
          + badge("✍️ build it", f"exercise {n:02d}", "6FE3B4", f"../../classroom/exercises/ex{n:02d}.py") + "\n"
          + badge("💬 ask", "the model", "FF6F61", f"https://github.com/{USER}/{REPO}/issues/new?template=ask-the-model.yml") + "\n</p>\n<!-- /DO -->")
    q = QUIZ[d.name]
    predict = (f"<!-- PREDICT -->\n> [!IMPORTANT]\n> **🎯 Predict before you read.** {q['predict']}\n>\n> Hold your answer in your head. You will check it at the bottom of the page.\n<!-- /PREDICT -->")
    if "<!-- PREDICT -->" not in text:
        text = text.replace("<!-- /DO -->", "<!-- /DO -->\n\n<!-- PREDICT -->\n<!-- /PREDICT -->", 1)
    text = re.sub(r"<!-- PREDICT -->.*?<!-- /PREDICT -->", lambda _: predict, text, flags=re.S)
    opts = "\n\n".join(
        f"<details><summary>{'ABC'[k]}. {o}</summary>\n\n"
        + (f"> ✅ **Yes.** {q['why']}" if k == q["answer"] else "> ❌ Not quite. Close this and try another one.") + "\n\n</details>"
        for k, o in enumerate(q["options"]))
    quiz = (f"<!-- QUIZ -->\n## Check yourself\n\n*Your prediction from the top of the page: was it right? Now this one.*\n\n**{q['question']}** &nbsp; Click an answer.\n\n{opts}\n\n"
            f"**Your turn:** open [`classroom/exercises/ex{n:02d}.py`](../../classroom/exercises/ex{n:02d}.py), press the pencil icon to edit it "
            f"right on GitHub (in your fork), fill in the function and commit; or run `python classroom/check.py` locally. [How the exercises work.](../../START_HERE.md#the-exercises-three-ways)\n<!-- /QUIZ -->")
    if "<!-- DO -->" not in text:
        text = text.replace("<!-- /NAV -->", "<!-- /NAV -->\n\n<!-- DO -->\n<!-- /DO -->", 1)
    if "<!-- QUIZ -->" not in text:
        k = text.rindex("\n---\n")
        text = text[:k] + "\n<!-- QUIZ -->\n<!-- /QUIZ -->\n" + text[k:]
    text = re.sub(r"<!-- NAV -->.*?<!-- /NAV -->", lambda _: nav, text, flags=re.S)
    text = re.sub(r"<!-- DO -->.*?<!-- /DO -->", lambda _: do, text, flags=re.S)
    text = re.sub(r"<!-- QUIZ -->.*?<!-- /QUIZ -->", lambda _: quiz, text, flags=re.S)

    out = subprocess.run([sys.executable, str(d / "run.py")], capture_output=True, text=True, cwd=ROOT)
    if out.returncode:
        sys.exit(f"{d.name}/run.py failed:\n{out.stderr}")
    body = ANSI.sub("", out.stdout).rstrip()
    block = (f"<!-- RUN:{d.name} -->\n<details open>\n<summary><b>Real output</b> from the model saved in this repository</summary>\n\n"
             f"```text\n{body}\n```\n\n</details>\n<!-- /RUN -->")
    text = re.sub(rf"<!-- RUN:{d.name} -->(.*?<!-- /RUN -->)?", lambda _: block, text, flags=re.S)
    readme.write_text(text)
    print(f"refreshed {readme.relative_to(ROOT)}")
