"""
Turns every stages/*/run.py into an executed Jupyter notebook in notebooks/. GitHub renders notebooks
with their output, so each lesson can be READ on github.com, and RUN with one click in Colab or Codespaces.
Run:  python tools/make_notebooks.py        (needs: pip install nbformat nbclient ipykernel)
"""
import json
import re
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
USER = "Normansrule"
REPO = "transparent-transformer-llm"
quiz = json.loads((ROOT / "classroom" / "quiz.json").read_text())

SETUP = f'''# Setup: works on GitHub Codespaces, on your own machine, and on Google Colab (where it clones the repository first).
import os, sys
if not os.path.exists("transparent_transformer"):
    if os.path.exists("../transparent_transformer"):
        os.chdir("..")
    else:
        os.system("git clone -q https://github.com/{USER}/{REPO}.git")
        os.chdir("{REPO}")
sys.path.insert(0, os.getcwd())
print("ready, working in", os.getcwd())'''

(ROOT / "notebooks").mkdir(exist_ok=True)
for d in sorted(p for p in (ROOT / "stages").iterdir() if p.is_dir()):
    src = (d / "run.py").read_text()
    doc = re.match(r'"""(.*?)"""', src, re.S).group(1).split("\nRun:")[0].strip()
    body = src[re.match(r'""".*?"""\n', src, re.S).end():]
    body = re.sub(r"^import sys\nfrom pathlib import Path\n\nsys\.path\.insert.*\n", "", body.lstrip("\n"), flags=re.M)
    body = re.sub(r'sys\.argv\[1\] if len\(sys\.argv\) > 1 else (".*?")', r"\1   # <- change this and re-run", body)
    body = re.sub(r'\nprint\("(\\n)?->  python stages.*', "", body)
    n = int(d.name[:2])
    q = quiz[d.name]
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"# Lesson {n}: {d.name.split('_', 1)[1].replace('_', ' ')}\n\n{doc}\n\n"
            f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{USER}/{REPO}/blob/main/notebooks/{d.name}.ipynb) &nbsp; "
            f"[lesson page](../stages/{d.name}/) &nbsp;|&nbsp; [live in your browser](https://{USER}.github.io/{REPO}/#{n})\n\n"
            "The output below was produced by the model saved in this repository. Run the cells to reproduce it, then change things."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(body.strip()),
        nbf.v4.new_markdown_cell(f"## Your turn\n\n**Think first:** {q['question']}\n\n"
                                 f"Then open `classroom/exercises/ex{n:02d}.py`, fill in the function, and run the cell below to grade it."),
        nbf.v4.new_code_cell("!python classroom/check.py | head -14"),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=120, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, ROOT / "notebooks" / f"{d.name}.ipynb")
    print("wrote notebooks/" + d.name + ".ipynb")
