"""
Builds every form of the flashcards from one source, flashcards/cards.json:
    docs/flashcards.js         the interactive flip-card page on the website
    flashcards/README.md       click-to-reveal cards readable right on GitHub
    flashcards/anki/*.tsv      import into Anki (File -> Import; fields: front, back)
Run:  python tools/make_flashcards.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER, REPO = "Normansrule", "transparent-transformer-llm"
SITE = f"https://{USER}.github.io/{REPO}"
cards = json.loads((ROOT / "flashcards" / "cards.json").read_text())
decks = list(dict.fromkeys(c["deck"] for c in cards))
slug = lambda d: re.sub(r"[^a-z0-9]+", "-", d.lower()).strip("-")                      # noqa: E731
ICON = {"the ten stages": "🔟", "training": "🏋️", "harness": "🧰", "perceptron and parameters": "🧠", "what is AI": "❓",
        "agents": "🤖", "coding agents": "💻", "history": "📜", "minds and machines": "🌌", "AI at work": "🏢", "beyond the transformer": "🚀", "claude and chatgpt": "⚖️", "looking inside": "🔬"}


def link(more):
    if not more:
        return ""
    if more.startswith("docs/"):
        return f"{SITE}/{more[5:]}"
    return f"https://github.com/{USER}/{REPO}/blob/main/{more}" if more.endswith(".md") else f"https://github.com/{USER}/{REPO}/tree/main/{more}"


(ROOT / "docs" / "flashcards.js").write_text("window.CARDS = " + json.dumps(
    [dict(c, link=link(c["more"]), icon=ICON.get(c["deck"], "📇")) for c in cards], ensure_ascii=False) + ";\n")

md = [f"# 📇 Flashcards\n\n<a href=\"{SITE}/flashcards.html\"><img src=\"../assets/flashcards.svg\" width=\"100%\" alt=\"A flashcard flipping from question to answer\"></a>\n",
      f"**{len(cards)} cards in {len(decks)} decks.** Three ways to study them:\n",
      f"| | how | best for |\n|---|---|---|\n| 🃏 | **[the flip-card page]({SITE}/flashcards.html)**: flip, grade yourself, it remembers what you know | real studying, on any device |\n"
      f"| 👇 | **right here**: click a question to reveal its answer | quick review on GitHub |\n"
      f"| 🗂️ | **Anki**: import a file from [`anki/`](anki/) (File → Import) | spaced repetition over weeks |\n",
      "| deck | cards | study |\n|---|:-:|:-:|"]
for d in decks:
    n = sum(c["deck"] == d for c in cards)
    md.append(f"| {ICON.get(d,'📇')} [{d}](#{slug(d)}) | {n} | [flip]({SITE}/flashcards.html#{slug(d)}) |")
md.append("")
for d in decks:
    md.append(f'<a id="{slug(d)}"></a>\n\n## {ICON.get(d,"📇")} {d}\n')
    for c in (c for c in cards if c["deck"] == d):
        more = f' &nbsp;<sub><a href="{link(c["more"])}">learn more</a></sub>' if c["more"] else ""
        md.append(f"<details><summary><b>{c['q']}</b></summary>\n\n> {c['a']}{more}\n\n</details>\n")
    md.append(f"<p align=\"right\"><a href=\"{SITE}/flashcards.html#{slug(d)}\">study this deck with flip cards →</a></p>\n")
(ROOT / "flashcards" / "README.md").write_text("\n".join(md))

(ROOT / "flashcards" / "anki").mkdir(exist_ok=True)
clean = lambda s: s.replace("\t", " ").replace("\n", " ")                          # noqa: E731
for d in decks:
    rows = [f"{clean(c['q'])}\t{clean(c['a'])}" for c in cards if c["deck"] == d]
    (ROOT / "flashcards" / "anki" / f"{slug(d)}.tsv").write_text("\n".join(rows) + "\n")
(ROOT / "flashcards" / "anki" / "all-decks.tsv").write_text("\n".join(f"{clean(c['q'])}\t{clean(c['a'])}\t{slug(c['deck'])}" for c in cards) + "\n")
print(f"{len(cards)} cards, {len(decks)} decks -> docs/flashcards.js, flashcards/README.md, flashcards/anki/")
