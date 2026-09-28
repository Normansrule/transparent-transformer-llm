"""
Draws the animated diagrams for the Understand pages (understand/), the flashcards and the README.
Run:  python tools/make_frames.py        (no model needed: these are concept diagrams)
Same design system and SMIL animation approach as tools/make_visuals.py.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_visuals import (AMBER, ASSETS, BG, CORAL, CYAN, GRID, INK, MINT, MONO, MUTED,  # noqa: E402
                          PANEL, SANS, W, appear, arrow, frame, text)

VIOLET = "#C9A7FF"


def box(x, y, w, h, fill=PANEL, stroke=MUTED, r=10, sw=1.6, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def lines(x, y, rows, size=13, fill=INK, font=SANS, anchor="start", gap=17, weight="400"):
    return "".join(text(x, y + i * gap, r, size, fill, font, anchor, weight) for i, r in enumerate(rows))


def glow(x, y, w, h, col, t0, t1, dur, r=10):
    """A coloured overlay that is visible between t0 and t1 of the loop."""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{col}" opacity="0">'
            f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" values="0;0;0.28;0.28;0;0" '
            f'keyTimes="0;{t0:.3f};{t0 + 0.02:.3f};{t1:.3f};{min(1, t1 + 0.02):.3f};1"/></rect>')


# ---------------------------------------------------------------- the course map
def course_map():
    b, dur = [], 12
    nodes = [("History", "1943 → 2026", 140, 120, VIOLET), ("What is AI?", "four approaches", 367, 120, CYAN),
             ("Agents", "percept → action", 594, 120, CYAN), ("Agent = harness + model", "stage 11", 821, 120, MINT),
             ("The transformer", "stages 1 to 5, 9, 10", 821, 260, AMBER), ("Training", "stages 6, 7, 8", 594, 260, AMBER),
             ("AI at work", "how it is used", 367, 260, CYAN), ("Minds and machines", "open questions", 140, 260, VIOLET)]
    for i, (a, bb) in enumerate(zip(nodes, nodes[1:])):
        b.append(arrow(a[2] + (78 if a[3] == bb[3] and a[2] < bb[2] else 0) if False else a[2], a[3], bb[2], bb[3], GRID, 2))
    for i, (t, s, x, y, col) in enumerate(nodes):
        t0 = 0.04 + 0.11 * i
        b.append(box(x - 100, y - 34, 200, 68, PANEL, col, 12, 2))
        b.append(glow(x - 100, y - 34, 200, 68, col, t0, t0 + 0.16, dur, 12))
        b.append(text(x, y - 4, t, 14 if len(t) > 16 else 15, INK, SANS, "middle", "700") + text(x, y + 17, s, 12, MUTED, SANS, "middle"))
    pts = ";".join(f"{x} {y}" for _, _, x, y, _ in nodes)
    kt = ";".join(f"{0.04 + 0.11 * i:.3f}" for i in range(len(nodes)))
    b.append(f'<circle r="7" fill="{AMBER}"><animateMotion dur="{dur}s" repeatCount="indefinite" calcMode="linear" '
             f'values="{nodes[0][2]} {nodes[0][3] - 44};' + ";".join(f"{x} {y - 44}" for _, _, x, y, _ in nodes) +
             f';{nodes[-1][2]} {nodes[-1][3] - 44}" keyTimes="0;{kt};1"/></circle>')
    b.append(text(W / 2, 64, "six frames of reference around one working model", 13, MUTED, SANS, "middle"))
    return frame(330, "".join(b), "understand  /  how the pieces of this course fit together")


# ---------------------------------------------------------------- four approaches to AI
def four_approaches():
    b, dur = [], 12
    cells = [("thinking humanly", "cognitive modelling", ["match the steps of human", "thought (Newell & Simon)"], 0, 0),
             ("thinking rationally", "laws of thought", ["correct reasoning in logic,", "from Aristotle onward"], 1, 0),
             ("acting humanly", "the Turing test", ["answers you cannot tell", "from a person's (1950)"], 0, 1),
             ("acting rationally", "rational agents", ["do the best expected thing:", "AIMA, and this repository"], 1, 1)]
    x0, y0, cw, ch = 250, 96, 330, 118
    b.append(text(x0 + cw / 2, 84, "like a human", 14, MUTED, SANS, "middle", "600") + text(x0 + cw * 1.5 + 12, 84, "rationally", 14, MUTED, SANS, "middle", "600"))
    b.append(text(x0 - 20, y0 + ch / 2 + 5, "thinking", 14, MUTED, SANS, "end", "600") + text(x0 - 20, y0 + ch * 1.5 + 17, "acting", 14, MUTED, SANS, "end", "600"))
    for i, (t, s, rows, cx, cy) in enumerate(cells):
        x, y = x0 + cx * (cw + 12), y0 + cy * (ch + 12)
        hot = i == 3
        b.append(box(x, y, cw, ch, PANEL, MINT if hot else GRID, 12, 2.4 if hot else 1.4))
        b.append(glow(x, y, cw, ch, MINT if hot else CYAN, 0.05 + 0.22 * i, 0.05 + 0.22 * i + 0.2, dur, 12))
        b.append(text(x + 18, y + 30, t, 17, MINT if hot else INK, SANS, weight="700") + text(x + 18, y + 50, s, 13, AMBER, SANS, weight="600"))
        b.append(lines(x + 18, y + 76, rows, 13, MUTED))
    b.append(text(40, 380, "Language models imitate human text (acting humanly). Wrapped in a harness and measured, they become rational agents.", 13, MUTED, SANS))
    return frame(400, "".join(b), "frame 1  /  four definitions of artificial intelligence")


# ---------------------------------------------------------------- the agent loop
def agent_loop():
    b, dur = [], 12
    b.append(box(60, 110, 250, 150, PANEL, CYAN, 14, 2) + text(185, 142, "ENVIRONMENT", 14, CYAN, SANS, "middle", "700"))
    b.append(box(650, 110, 250, 150, PANEL, AMBER, 14, 2) + text(775, 142, "AGENT", 14, AMBER, SANS, "middle", "700"))
    b.append(f'<path d="M310 150 C 470 90, 490 90, 650 150" fill="none" stroke="{MUTED}" stroke-width="2" marker-end="url(#m)"/>')
    b.append(f'<path d="M650 225 C 490 285, 470 285, 310 225" fill="none" stroke="{MUTED}" stroke-width="2"/>')
    b.append(text(480, 96, "percepts, through sensors", 13, INK, SANS, "middle", "600") + text(480, 300, "actions, through actuators", 13, INK, SANS, "middle", "600"))
    b.append(f'<circle r="7" fill="{CYAN}"><animateMotion dur="3s" repeatCount="indefinite" path="M310 150 C 470 90, 490 90, 650 150"/></circle>')
    b.append(f'<circle r="7" fill="{AMBER}"><animateMotion dur="3s" begin="1.5s" repeatCount="indefinite" path="M650 225 C 490 285, 470 285, 310 225"/></circle>')
    views = [("a vacuum robot", ["two squares, some dirt"], ["location, dirt sensor"], ["if dirty: suck,", "else move"], ["move, suck"]),
             ("this repository's weather agent", ["you; the Open-Meteo service"], ["your message; tool results"], ["harness.py around", "the transformer"], ["reply; get_weather()"]),
             ("a coding agent", ["a code repository; a shell"], ["request; file contents"], ["harness loop around", "a language model"], ["read, write, search, run"])]
    n = len(views)
    for k, (name, env, sens, prog, act) in enumerate(views):
        g = [text(W / 2, 64, f"example {k + 1} of {n}: {name}", 14, INK, SANS, "middle", "700"),
             lines(185, 176, env, 13, INK, SANS, "middle"), lines(775, 172, prog, 13, INK, SANS, "middle"),
             text(480, 116, sens[0], 12.5, CYAN, MONO, "middle"), text(480, 322, act[0], 12.5, AMBER, MONO, "middle")]
        b.append(f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" begin="{k * dur / n}s" repeatCount="indefinite" '
                 f'values="0;1;1;0;0" keyTimes="0;0.02;{1 / n - 0.02:.3f};{1 / n:.3f};1"/>{"".join(g)}</g>')
    return frame(350, "".join(b), "frame 2  /  an agent perceives and acts")


def agent_equation():
    rows = [("AIMA", "agent", "architecture", "program", "the machine with its sensors", "decides what to do", CYAN),
            ("the coding-agent course", "agent", "harness", "model", "ordinary code: prompts, tools, stopping", "predicts the next token", MINT),
            ("one level down", "model", "architecture", "weights", "code: transformer.py", "learned numbers: artifacts/*.npz", AMBER)]
    b = []
    for i, (who, lhs, a, c, da, dc, col) in enumerate(rows):
        y, t0 = 92 + i * 92, 0.06 + 0.26 * i
        g = [text(40, y + 6, who, 12.5, MUTED, SANS), text(250, y + 12, lhs, 24, col, MONO, "middle", "700"),
             text(300, y + 12, "=", 24, INK, MONO, "middle"), box(330, y - 22, 220, 48, PANEL, col, 10, 2), text(440, y + 9, a, 18, INK, MONO, "middle", "600"),
             text(575, y + 12, "+", 24, INK, MONO, "middle"), box(600, y - 22, 220, 48, PANEL, col, 10, 2), text(710, y + 9, c, 18, INK, MONO, "middle", "600"),
             text(440, y + 44, da, 11.5, MUTED, SANS, "middle"), text(710, y + 44, dc, 11.5, MUTED, SANS, "middle")]
        b.append(f'<g opacity="0">{appear(t0, 0.97, 9)}{"".join(g)}</g>')
    return frame(360, "".join(b), "frame 2  /  the same idea at three levels")


def agent_types():
    types = [("simple reflex", "if-then rules on the current percept", "the input guard's blocklist"),
             ("model-based reflex", "rules plus a model of what it cannot see", "memory of earlier turns"),
             ("goal-based", "which action reaches the goal?", "the router calls the weather tool"),
             ("utility-based", "which outcome is best overall?", "the output guard prefers correct numbers"),
             ("learning", "improves its own parts from feedback", "pretraining, SFT and DPO")]
    b = []
    for i, (t, how, here) in enumerate(types):
        y, x, t0 = 330 - i * 58, 60 + i * 60, 0.05 + 0.15 * i
        col = [CORAL, AMBER, CYAN, MINT, VIOLET][i]
        g = [box(x, y - 22, 300, 46, PANEL, col, 10, 2), text(x + 16, y + 6, t, 16, col, SANS, weight="700"),
             text(x + 320, y - 2, how, 12.5, INK, SANS), text(x + 320, y + 15, "here: " + here, 12.5, MUTED, SANS)]
        b.append(f'<g opacity="0">{appear(t0, 0.97, 10)}{"".join(g)}</g>')
    b.append(text(40, 70, "each rung can do everything the rung below can, and more", 13, MUTED, SANS))
    return frame(380, "".join(b), "frame 2  /  five kinds of agent program")


# ---------------------------------------------------------------- coding agents
def six_components():
    comps = ["live repo context", "prompt shape + cache reuse", "tool access and use", "minimising context bloat", "structured session memory", "bounded delegation"]
    b, cx, cy, dur = [], W / 2, 215, 12
    spots = [(cx + math.cos(-math.pi / 2 + i * math.pi / 3) * 320, cy + math.sin(-math.pi / 2 + i * math.pi / 3) * 145) for i in range(6)]
    for x, y in spots:
        b.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.0f}" y2="{y:.0f}" stroke="{GRID}" stroke-width="2"/>')
    b.append(box(cx - 90, cy - 40, 180, 80, PANEL, AMBER, 14, 2.4) + text(cx, cy - 4, "the model", 17, INK, SANS, "middle", "700") + text(cx, cy + 18, "text in, text out", 12, MUTED, SANS, "middle"))
    for i, (c, (x, y)) in enumerate(zip(comps, spots)):
        t0 = 0.04 + 0.14 * i
        b.append(box(x - 125, y - 24, 250, 48, PANEL, CYAN, 10, 1.8) + glow(x - 125, y - 24, 250, 48, MINT, t0, t0 + 0.14, dur))
        b.append(text(x - 108, y + 6, str(i + 1), 16, AMBER, MONO, weight="700") + text(x + 10, y + 6, c, 13, INK, SANS, "middle", "600"))
    b.append(text(40, 402, "Sebastian Raschka's six components: all harness, none of them training", 13, MUTED, SANS))
    return frame(420, "".join(b), "frame 3  /  the components of a coding agent")


def two_datasets():
    b = []
    for k, (title, note, tokens, col, tests) in enumerate([
            ("instruction pairs", "loss on the reply only", [("prompt: rules, tools, transcript, request", False), ("<tool>{...}</tool>", True)], MINT, "tests: it needs the FORMAT"),
            ("raw code corpus", "loss on every token", [("def bubble_sort(items):", True), ("    values = list(items) ...", True)], CYAN, "tests: it needs more CODE")]):
        x = 40 + k * 460
        b.append(box(x, 80, 420, 190, PANEL, col, 14, 2) + text(x + 20, 110, title, 17, col, SANS, weight="700") + text(x + 20, 130, note, 12.5, MUTED, SANS))
        for j, (tok, graded) in enumerate(tokens):
            y, t0 = 160 + j * 44, 0.1 + 0.2 * j + 0.05 * k
            w = min(380, len(tok) * 8.2 + 20)
            b.append(f'<g opacity="0">{appear(t0, 0.97, 8)}{box(x + 20, y - 18, w, 30, (col if graded else GRID), "none", 6, 0)}'
                     + text(x + 30, y + 2, tok, 12.5, "#0B1E33" if graded else MUTED, MONO) + "</g>")
        b.append(text(x + 20, 256, tests, 13, INK, SANS, weight="600"))
    b.append(text(40, 300, "coloured = graded by the loss. The same idea as the loss mask in stage 8.", 13, MUTED, SANS))
    return frame(320, "".join(b), "frame 3  /  the GPT-2 agent project's two datasets")


# ---------------------------------------------------------------- history
EVENTS = [(1943, "artificial neuron", VIOLET), (1950, "Turing test", CYAN), (1956, "'AI' named", CYAN), (1958, "perceptron", VIOLET),
          (1969, "Perceptrons critique", CORAL), (1971, "Intel 4004", MUTED), (1986, "backprop", VIOLET), (1989, "Emperor's New Mind", AMBER),
          (1995, "AIMA", CYAN), (1996, "Orch OR", AMBER), (1997, "Deep Blue", CYAN), (2004, "IIT", AMBER), (2012, "AlexNet", VIOLET),
          (2017, "transformer", MINT), (2019, "GPT-2", MINT), (2020, "GPT-3", MINT), (2022, "ChatGPT", MINT), (2026, "86% of studios", CORAL)]


def timeline():
    b, x0, x1, y = [], 60, W - 60, 230
    X = lambda yr: x0 + (x1 - x0) * (yr - 1940) / (2028 - 1940)                                   # noqa: E731
    b.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{GRID}" stroke-width="5" stroke-linecap="round"/>')
    for yr in range(1940, 2031, 10):
        b.append(text(X(yr), y + 30, str(yr), 11, MUTED, MONO, "middle"))
    for i, (yr, label, col) in enumerate(EVENTS):
        up = i % 2 == 0
        lvl = (i // 2) % 4
        ly = y - 34 - lvl * 32 if up else y + 54 + lvl * 26
        t0 = 0.03 + 0.045 * i
        g = [f'<line x1="{X(yr):.0f}" y1="{y}" x2="{X(yr):.0f}" y2="{ly + (8 if up else -14)}" stroke="{col}" stroke-width="1.4"/>',
             f'<circle cx="{X(yr):.0f}" cy="{y}" r="6" fill="{col}"/>', text(X(yr), ly, f"{yr} {label}", 11.5, col, SANS, "middle", "600")]
        b.append(f'<g opacity="0">{appear(t0, 0.97, 14)}{"".join(g)}</g>')
    legend = [(VIOLET, "neural networks"), (CYAN, "AI as a field"), (AMBER, "theories of mind"), (MINT, "language models"), (CORAL, "turning points")]
    for i, (c, l) in enumerate(legend):
        b.append(f'<circle cx="{60 + i * 170}" cy="{y + 180}" r="6" fill="{c}"/>' + text(72 + i * 170, y + 185, l, 12, MUTED, SANS))
    return frame(y + 200, "".join(b), "frame 4  /  from an artificial neuron to AI in every studio")


# ---------------------------------------------------------------- minds and machines
def theories_of_mind():
    cols = [("computational", "Baars, Dehaene, Dennett", "the right organisation of", "information processing", ("not now; a future", "design could be"), CYAN),
            ("IIT", "Tononi, Koch", "integrated information,", "Φ, in the causal structure", ("no: feedforward,", "so Φ = 0"), MINT),
            ("Orch OR", "Penrose, Hameroff", "quantum state reductions", "in neuron microtubules", ("no: it is classical", "computation"), AMBER),
            ("fundamental", "Faggin", "a basic feature of reality,", "tied to quantum information", ("no: copyable and", "classical"), VIOLET)]
    b = []
    for i, (name, who, c1, c2, verdict, col) in enumerate(cols):
        x, t0 = 40 + i * 222, 0.05 + 0.2 * i
        g = [box(x, 80, 206, 250, PANEL, col, 14, 2), text(x + 103, 112, name, 18, col, SANS, "middle", "700"), text(x + 103, 132, who, 12, MUTED, SANS, "middle"),
             text(x + 16, 170, "consciousness is", 11.5, MUTED, SANS), lines(x + 16, 190, [c1, c2], 13, INK), f'<line x1="{x + 16}" y1="{236}" x2="{x + 190}" y2="236" stroke="{GRID}"/>',
             text(x + 16, 258, "could this transformer be?", 11.5, MUTED, SANS), lines(x + 16, 280, list(verdict), 12.5, col, SANS, gap=18, weight="600")]
        b.append(f'<g opacity="0">{appear(t0, 0.97, 10)}{"".join(g)}</g>')
    b.append(text(40, 356, "Every column has serious critics. Read frame 5 for the objections to each.", 13, MUTED, SANS))
    return frame(376, "".join(b), "frame 5  /  four answers to 'could a machine be conscious?'")


def compute_scales():
    rows = [("a laptop processor", 5e10, "floating-point operations per second", CYAN), ("one datacenter GPU", 1e15, "operations per second, low precision", CYAN),
            ("the brain, counting synapses", 4e15, "bits per second (Moravec, cited in the 1990 paper)", MINT),
            ("the brain, counting microtubules", 1e24, "bits per second (Rasmussen, Hameroff et al. 1990: 10²³ to 10²⁵)", AMBER)]
    b, x0, bw, lo, hi = [], 300, 560, 9, 26
    for e in range(10, 27, 2):
        x = x0 + bw * (e - lo) / (hi - lo)
        b.append(f'<line x1="{x:.0f}" y1="84" x2="{x:.0f}" y2="{84 + len(rows) * 58}" stroke="{GRID}"/>' + text(x, 100 + len(rows) * 58, f"10^{e}", 11, MUTED, MONO, "middle"))
    for i, (name, v, unit, col) in enumerate(rows):
        y, w = 92 + i * 58, bw * (math.log10(v) - lo) / (hi - lo)
        b.append(text(x0 - 12, y + 18, name, 14, INK, SANS, "end"))
        b.append(f'<rect x="{x0}" y="{y}" width="{w:.0f}" height="26" rx="4" fill="{col}"><animate attributeName="width" values="0;{w:.0f};{w:.0f}" keyTimes="0;0.3;1" dur="7s" begin="{i * 0.3}s" fill="freeze"/></rect>')
        b.append(text(x0, y + 42, unit, 11.5, MUTED, SANS))
    b.append(text(40, 70, "rough estimates in different units: read them as orders of magnitude, not as a race", 13, MUTED, SANS))
    return frame(150 + len(rows) * 58, "".join(b), "frame 5  /  how much computing? four very different estimates")


# ---------------------------------------------------------------- AI at work
def adoption():
    b = []
    bars = [("2025", 0.51, MUTED, "51%"), ("2026", 0.858, MINT, "85.8%")]
    for i, (yr, v, col, lab) in enumerate(bars):
        y, w = 110 + i * 70, 600 * v
        b.append(text(150, y + 30, yr, 20, INK, MONO, "end", "700") + box(170, y, 600, 44, PANEL, GRID, 8, 1))
        b.append(f'<rect x="170" y="{y}" width="{w:.0f}" height="44" rx="8" fill="{col}"><animate attributeName="width" values="0;{w:.0f};{w:.0f}" keyTimes="0;0.35;1" dur="7s" begin="{i * 0.6}s" fill="freeze"/></rect>')
        b.append(text(190 + w, y + 30, lab, 20, INK, MONO, weight="700"))
    b.append(text(170, 280, "of Japanese game developers surveyed use generative AI; 63% use it daily", 14, INK, SANS))
    b.append(text(170, 304, "Mostly debugging, error fixing and automated testing, always supervised by people.", 12.5, MUTED, SANS))
    b.append(text(170, 324, "Source: CESA survey preview, Tokyo Game Show 2026, as reported by TechSpot.", 12, MUTED, SANS))
    return frame(350, "".join(b), "frame 6  /  AI at work in one industry")


def flashcards_banner():
    b, dur = [], 6
    cx, cy, w, h = W / 2, 160, 460, 190
    front = box(cx - w / 2, cy - h / 2, w, h, PANEL, CYAN, 18, 2) + text(cx, cy - 36, "🔟 THE TEN STAGES", 12, AMBER, MONO, "middle", "600") + lines(cx, cy - 2, ["What is the residual stream?"], 22, INK, SANS, "middle", 24, "700") + text(cx, cy + 62, "click to flip", 12, MUTED, SANS, "middle")
    back = box(cx - w / 2, cy - h / 2, w, h, "#123f38", MINT, 18, 2) + text(cx, cy - 46, "ANSWER", 12, MINT, MONO, "middle", "600") + lines(cx, cy - 12, ["The shared channel of token vectors.", "Every block reads it and ADDS", "its result; nothing overwrites it."], 16, INK, SANS, "middle", 22)
    anim = lambda a, b2: f'<animateTransform attributeName="transform" type="scale" additive="sum" dur="{dur}s" repeatCount="indefinite" values="{a}" keyTimes="{b2}"/>'  # noqa: E731
    b.append(f'<g transform="translate({cx} 0)"><g><g transform="translate({-cx} 0)">{front}</g>{anim("1 1;1 1;0 1;0 1;0 1;1 1", "0;0.35;0.45;0.5;0.9;1")}</g></g>')
    b.append(f'<g transform="translate({cx} 0)"><g><g transform="translate({-cx} 0)">{back}</g>{anim("0 1;0 1;0 1;1 1;1 1;0 1", "0;0.35;0.45;0.55;0.9;1")}</g></g>')
    for i, (lab, col) in enumerate([("✗ again", CORAL), ("✓ knew it", MINT)]):
        b.append(box(cx - 170 + i * 180, 280, 160, 40, col, "none", 10, 0) + text(cx - 90 + i * 180, 306, lab, 15, "#0B1E33", SANS, "middle", "700"))
    b.append(text(40, 70, "113 cards, 10 decks: on the website, readable on GitHub, or in Anki", 13, MUTED, SANS))
    return frame(350, "".join(b), "flashcards  /  every idea in the course, one card at a time")


def main():
    out = {"course_map": course_map(), "ai_four_approaches": four_approaches(), "agent_loop": agent_loop(), "agent_equation": agent_equation(),
           "agent_types": agent_types(), "six_components": six_components(), "two_datasets": two_datasets(), "timeline": timeline(),
           "theories_of_mind": theories_of_mind(), "compute_scales": compute_scales(), "adoption": adoption(), "flashcards": flashcards_banner()}
    for k, v in out.items():
        (ASSETS / f"{k}.svg").write_text(v)
    print(f"wrote {len(out)} concept diagrams -> assets/")


if __name__ == "__main__":
    main()
