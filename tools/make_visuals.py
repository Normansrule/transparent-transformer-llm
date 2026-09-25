"""
Draws every animated diagram in this repository from the model's REAL numbers.
Run:  python tools/make_visuals.py      (after `python -m transparent_transformer.trace`)

The pictures are plain Scalable Vector Graphics (SVG) files animated with SMIL
(Synchronized Multimedia Integration Language) tags such as <animate>. GitHub
shows them, moving, inside any README. No JavaScript, no GIFs.
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets"

# ------------------------------------------------------------------ design tokens
BG, GRID, INK, MUTED = "#0F2A47", "#1B3F66", "#DCE9F5", "#7FA3C7"
AMBER, CORAL, MINT, CYAN, PANEL = "#FFB238", "#FF6F61", "#6FE3B4", "#5CC8FF", "#143556"
CHIPS = [AMBER, CYAN, MINT, "#C9A7FF", CORAL, "#9AD0FF"]
MONO = "ui-monospace,'SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace"
SANS = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
W = 960

STAGES = ["Input", "Tokens", "Embed", "Transformer", "Attention",
          "Pretrain", "Backprop", "Align", "Sample", "Output"]


def frame(h: int, body: str, title: str = "") -> str:
    head = (f'<text x="28" y="34" font-family="{SANS}" font-size="14" fill="{MUTED}" '
            f'letter-spacing="0.4">{escape(title)}</text>') if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img">'
            f'<defs><pattern id="g" width="24" height="24" patternUnits="userSpaceOnUse">'
            f'<path d="M24 0H0V24" fill="none" stroke="{GRID}" stroke-width="1"/></pattern></defs>'
            f'<rect width="{W}" height="{h}" rx="14" fill="{BG}"/>'
            f'<rect width="{W}" height="{h}" rx="14" fill="url(#g)" opacity="0.55"/>'
            f'{head}{body}</svg>')


def clean(s: str) -> str:
    """XML cannot hold control characters (an untrained model happily emits them)."""
    return "".join(ch if ch.isprintable() else "\u25a1" for ch in s)


def text(x, y, s, size=15, fill=INK, font=MONO, anchor="start", weight="400", extra=""):
    s = clean(s)
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" xml:space="preserve" {extra}>{escape(s)}</text>')


def show(s: str) -> str:
    """Make whitespace visible inside a token."""
    return s.replace(" ", "·").replace("\n", "↵")


def chip_w(s: str, size=15) -> float:
    return len(show(s)) * size * 0.602 + 14


def chip(x, y, s, colour, size=15, inner=""):
    w = chip_w(s, size)
    return (f'<g>{inner}<rect x="{x:.1f}" y="{y - size - 4:.1f}" width="{w:.1f}" height="{size + 12}" rx="5" '
            f'fill="{colour}"/>' + text(x + 7, y + 1, show(s), size, "#0B1E33", weight="600") + "</g>", w)


def appear(t0: float, t1: float = 0.96, dur: float = 10) -> str:
    """Invisible until fraction t0 of the loop, visible until t1, then reset."""
    a = max(t0 - 0.02, 0.0)
    return (f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" '
            f'values="0;0;1;1;0;0" keyTimes="0;{a:.3f};{t0:.3f};{t1:.3f};{min(t1 + 0.02, 1):.3f};1"/>')


def arrow(x1, y1, x2, y2, colour=MUTED, width=1.6, dash=""):
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    ax, ay = x2 - 9 * math.cos(ang), y2 - 9 * math.sin(ang)
    px, py = 4.5 * math.sin(ang), -4.5 * math.cos(ang)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{colour}" stroke-width="{width}"{d}/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {ax + px:.1f},{ay + py:.1f} {ax - px:.1f},{ay - py:.1f}" fill="{colour}"/>')


def heat(v: float, vmax: float) -> str:
    """Diverging colour: coral for negative, cyan for positive."""
    t = max(-1.0, min(1.0, v / vmax))
    base = (0x14, 0x35, 0x56)
    tgt = (0x5C, 0xC8, 0xFF) if t >= 0 else (0xFF, 0x6F, 0x61)
    a = abs(t) ** 0.7
    return "#" + "".join(f"{int(b + (c - b) * a):02X}" for b, c in zip(base, tgt))


# ------------------------------------------------------------------ pipeline strip
def station_x(i: int) -> float:
    return 62 + i * (W - 124) / 9


def pipeline(current: int) -> str:
    """The rail that appears at the top of every stage page. `current` is 1-based."""
    y, b = 78, []
    b.append(f'<line x1="{station_x(0)}" y1="{y}" x2="{station_x(9)}" y2="{y}" stroke="{GRID}" stroke-width="6" stroke-linecap="round"/>')
    b.append(f'<line x1="{station_x(0)}" y1="{y}" x2="{station_x(current - 1)}" y2="{y}" stroke="{MINT}" stroke-width="2" opacity="0.7"/>')
    # brackets: running the model vs. building the model
    for lo, hi, label in [(0, 4, "what happens when you press Enter"), (5, 7, "how the weights got their values"),
                          (8, 9, "...and back to your prompt")]:
        x1, x2 = station_x(lo) - 22, station_x(hi) + 22
        b.append(f'<path d="M{x1} 40v-7h{x2 - x1}v7" fill="none" stroke="{MUTED}" stroke-width="1" opacity="0.6"/>')
        b.append(text((x1 + x2) / 2, 25, label, 11.5, MUTED, SANS, "middle"))
    for i, name in enumerate(STAGES):
        x, n = station_x(i), i + 1
        if n < current:
            ring, fill, ink = MINT, BG, MINT
        elif n == current:
            ring, fill, ink = AMBER, AMBER, "#0B1E33"
        else:
            ring, fill, ink = MUTED, BG, MUTED
        pulse = (f'<circle cx="{x}" cy="{y}" r="15" fill="none" stroke="{AMBER}" stroke-width="2">'
                 f'<animate attributeName="r" values="15;15;27" keyTimes="0;0.55;1" dur="3.2s" repeatCount="indefinite"/>'
                 f'<animate attributeName="opacity" values="0;0.9;0" keyTimes="0;0.55;1" dur="3.2s" repeatCount="indefinite"/></circle>'
                 ) if n == current else ""
        b.append(f'{pulse}<circle cx="{x}" cy="{y}" r="15" fill="{fill}" stroke="{ring}" stroke-width="2"/>')
        b.append(text(x, y + 5, str(n), 13, ink, MONO, "middle", "700"))
        b.append(text(x, y + 38, name, 12.5, INK if n == current else MUTED, SANS, "middle", "600" if n == current else "400"))
    # the carriage: grabs the data packet at the previous station, carries it over, sets it down
    x0 = station_x(max(current - 2, 0)) if current > 1 else 8
    x1 = station_x(current - 1)
    b.append(f'<g><animateTransform attributeName="transform" type="translate" dur="3.2s" repeatCount="indefinite" '
             f'calcMode="spline" keySplines="0 0 1 1;0.6 0 0.3 1;0 0 1 1;0 0 1 1" keyTimes="0;0.12;0.55;0.9;1" '
             f'values="{x0} 0;{x0} 0;{x1} 0;{x1} 0;{x1} 0"/>'
             f'<path d="M-9 {y - 31}h18M0 {y - 31}v7M-8 {y - 17}v-7h16v7" fill="none" stroke="{AMBER}" stroke-width="2" stroke-linecap="round"/>'
             f'<rect x="-6" y="{y - 22}" width="12" height="9" rx="2" fill="{AMBER}">'
             f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.62;0.7;1" dur="3.2s" repeatCount="indefinite"/></rect></g>')
    return frame(134, "".join(b))


# ------------------------------------------------------------------ hero
def hero(tr: dict) -> str:
    b, y, dur = [], 176, 14
    prompt, answer = tr["prompt"], tr["output"]
    b.append(text(W / 2, 58, "transparent-transformer-llm", 32, INK, SANS, "middle", "700"))
    b.append(text(W / 2, 86, "a transformer you can see straight through: one prompt, ten stages, every number on show", 15, MUTED, SANS, "middle"))
    b.append(f'<line x1="{station_x(0)}" y1="{y}" x2="{station_x(9)}" y2="{y}" stroke="{GRID}" stroke-width="6" stroke-linecap="round"/>')
    n = len(STAGES)
    t_start, t_end = 0.12, 0.72                       # the packet travels during this slice of the loop
    for i, name in enumerate(STAGES):
        x = station_x(i)
        t_hit = t_start + (t_end - t_start) * i / (n - 1)
        training = 5 <= i <= 7
        col = CORAL if training else AMBER
        b.append(f'<circle cx="{x}" cy="{y}" r="16" fill="{BG}" stroke="{MUTED}" stroke-width="2"/>')
        b.append(f'<circle cx="{x}" cy="{y}" r="16" fill="{col}" opacity="0">'
                 f'<animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" values="0;0;1;0.25;0.25;0" '
                 f'keyTimes="0;{t_hit - 0.01:.3f};{t_hit:.3f};{t_hit + 0.06:.3f};0.95;1"/></circle>')
        b.append(text(x, y + 5, str(i + 1), 13, INK, MONO, "middle", "700"))
        b.append(text(x, y + 40, name, 12.5, MUTED, SANS, "middle"))
    xs = ";".join(f"{station_x(i):.0f} 0" for i in range(n))
    kt = ";".join(f"{t_start + (t_end - t_start) * i / (n - 1):.3f}" for i in range(n))
    b.append(f'<g opacity="0"><animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" '
             f'values="0;0;1;1;0;0" keyTimes="0;{t_start - 0.02};{t_start};{t_end};{t_end + 0.02};1"/>'
             f'<animateTransform attributeName="transform" type="translate" dur="{dur}s" repeatCount="indefinite" '
             f'calcMode="spline" keySplines="{";".join(["0 0 1 1"] + ["0.7 0 0.3 1"] * (n - 1) + ["0 0 1 1"])}" '
             f'values="{station_x(0):.0f} 0;{xs};{station_x(9):.0f} 0" keyTimes="0;{kt};1"/>'
             f'<path d="M-10 {y - 36}h20M0 {y - 36}v8M-9 {y - 20}v-8h18v8" fill="none" stroke="{AMBER}" stroke-width="2.2" stroke-linecap="round"/>'
             f'<rect x="-7" y="{y - 26}" width="14" height="10" rx="2" fill="{AMBER}"/></g>')
    x1, x2 = station_x(5) - 24, station_x(7) + 24
    b.append(f'<path d="M{x1} {y - 50}v-7h{x2 - x1}v7" fill="none" stroke="{CORAL}" stroke-width="1" opacity="0.8"/>')
    b.append(text((x1 + x2) / 2, y - 64, "the workshop: where the weights are made", 11.5, CORAL, SANS, "middle"))
    # prompt in, answer out
    b.append(text(40, 272, "in ", 14, MUTED, SANS))
    b.append(f'<g>{appear(0.02, 0.96, dur)}' + text(72, 272, prompt, 17, AMBER, MONO, weight="600") + "</g>")
    b.append(text(40, 306, "out", 14, MUTED, SANS))
    pieces = [g["piece"] for g in tr["generation"] if g["piece"] != "<|end|>"]
    x = 72.0
    for i, p in enumerate(pieces):
        t0 = t_end + 0.015 + 0.2 * i / len(pieces)
        p = p.lstrip(" ") if i == 0 else p
        b.append(f'<g opacity="0">{appear(t0, 0.96, dur)}' + text(x, 306, p, 17, MINT, MONO, weight="600") + "</g>")
        x += len(p) * 17 * 0.602
    return frame(336, "".join(b))


# ------------------------------------------------------------------ stage pictures
def s01_input(tr):
    b, s = [], tr["prompt"]
    x0, cw = 60, 23.2
    b.append(text(x0, 92, "what you typed", 13, MUTED, SANS))
    b.append(text(x0, 190, "what the computer stores: one number (byte) per character", 13, MUTED, SANS))
    for i, ch in enumerate(s):
        t0 = 0.04 + 0.5 * i / len(s)
        x = x0 + i * cw
        b.append(f'<g opacity="0">{appear(t0)}<rect x="{x}" y="104" width="{cw - 3}" height="34" rx="4" fill="{PANEL}" stroke="{GRID}"/>'
                 + text(x + (cw - 3) / 2, 127, show(ch), 17, AMBER, MONO, "middle", "600") + "</g>")
        b.append(f'<g opacity="0">{appear(t0 + 0.06)}' + f'<line x1="{x + 10}" y1="140" x2="{x + 10}" y2="200" stroke="{GRID}" stroke-width="1"/>'
                 + text(x + (cw - 3) / 2, 222, str(ord(ch)), 10.5, CYAN, MONO, "middle") + "</g>")
    b.append(f'<g opacity="0">{appear(0.68)}' + text(x0, 286, f"{len(s)} characters. To the machine this is only a row of numbers between 0 and 255.", 15, INK, SANS)
             + text(x0, 312, "The model cannot use them yet: single letters carry almost no meaning.", 15, MUTED, SANS) + "</g>")
    return frame(340, "".join(b), "stage 1  /  input")


def s02_tokens(tr):
    b = []
    steps = tr["tokens"]["merge_demo"]["steps"]
    b.append(text(60, 78, "Byte Pair Encoding (BPE): glue the most common neighbours together, again and again", 13, MUTED, SANS))
    n = len(steps)
    for k, row in enumerate(steps):
        t0, t1 = 0.03 + 0.5 * k / n, (0.03 + 0.5 * (k + 1) / n if k < n - 1 else 0.96)
        x, g = 60.0, []
        for i, p in enumerate(row):
            c, w = chip(x, 124, p, CHIPS[i % len(CHIPS)], 20)
            g.append(c)
            x += w + 6
        g.append(text(x + 14, 124, f"merge {k}" if k else "raw bytes", 13, MUTED, SANS))
        b.append(f'<g opacity="0">{appear(t0, t1)}{"".join(g)}</g>')
    b.append(text(60, 186, "the whole prompt, after the chat template is added", 13, MUTED, SANS))
    x, yy = 60.0, 226
    for i, (p, tid) in enumerate(zip(tr["tokens"]["pieces"], tr["tokens"]["ids"])):
        w = chip_w(p, 15)
        if x + w > W - 50:
            x, yy = 60.0, yy + 62
        t0 = 0.56 + 0.3 * i / len(tr["tokens"]["ids"])
        c, _ = chip(x, yy, p, CHIPS[i % len(CHIPS)], 15)
        b.append(f'<g opacity="0">{appear(t0)}{c}' + text(x + w / 2, yy + 24, str(tid), 12, CYAN, MONO, "middle") + "</g>")
        x += w + 6
    return frame(yy + 56, "".join(b), "stage 2  /  tokenization   (· marks a space that belongs to the token)")


def s03_embed(tr):
    b, e = [], tr["embedding"]
    pieces, ids = tr["tokens"]["pieces"], tr["tokens"]["ids"]
    rows = list(range(1, min(6, len(ids))))
    vmax = max(abs(v) for r in e["sum"] for v in r)
    cw, x_tab = 27, 440
    b.append(text(60, 76, "token", 13, MUTED, SANS) + text(190, 76, "id", 13, MUTED, SANS)
             + text(x_tab, 76, f"its vector: first 16 of {e['shape'][1]} learned numbers (coral = negative, blue = positive)", 13, MUTED, SANS))
    for k, r in enumerate(rows):
        y, t0 = 112 + k * 44, 0.05 + 0.75 * k / len(rows)
        c, w = chip(60, y, pieces[r], CHIPS[r % len(CHIPS)], 15)
        b.append(c + text(190, y, str(ids[r]), 15, CYAN, MONO, weight="600"))
        b.append(f'<g opacity="0">{appear(t0)}' + arrow(232, y - 5, x_tab - 118, y - 5, AMBER, 2)
                 + text(x_tab - 108, y, f"row {ids[r]}", 13, AMBER, MONO) + arrow(x_tab - 40, y - 5, x_tab - 10, y - 5, AMBER, 2) + "</g>")
        cells = "".join(f'<rect x="{x_tab + j * cw}" y="{y - 20}" width="{cw - 3}" height="28" rx="3" fill="{heat(v, vmax)}"/>'
                        for j, v in enumerate(e["sum"][r]))
        b.append(f'<g opacity="0">{appear(t0 + 0.08)}{cells}</g>')
    yb = 112 + len(rows) * 44 + 6
    b.append(text(60, yb + 14, "vector = token_table[id] + position_table[where it sits in the sentence]", 14, INK, MONO))
    return frame(yb + 40, "".join(b), "stage 3  /  embedding   (real values from the trained model)")


def s04_transformer(tr):
    b, y = [], 190
    cfg = tr["config"]
    b.append(f'<line x1="70" y1="{y}" x2="890" y2="{y}" stroke="{AMBER}" stroke-width="4" stroke-linecap="round" opacity="0.85"/>')
    b.append(text(40, y + 44, "the residual stream (amber line)", 12.5, AMBER, SANS))
    b.append(f'<rect x="40" y="{y - 22}" width="78" height="44" rx="6" fill="{PANEL}" stroke="{INK}"/>' + text(79, y + 5, "embed", 13, INK, SANS, "middle"))
    x = 150
    for li in range(cfg["n_layers"]):
        b.append(f'<rect x="{x - 12}" y="64" width="306" height="{y - 28}" rx="10" fill="none" stroke="{MUTED}" stroke-dasharray="5 5"/>')
        b.append(text(x, 86, f"block {li + 1}", 13, MUTED, SANS))
        for j, (name, col) in enumerate([("attention", CYAN), ("MLP", MINT)]):
            bx = x + 14 + j * 146
            t0 = 0.08 + 0.2 * (li * 2 + j)
            b.append(f'<path d="M{bx} {y}V128h20" fill="none" stroke="{col}" stroke-width="1.6"/>')
            b.append(f'<rect x="{bx + 20}" y="108" width="84" height="40" rx="6" fill="{PANEL}" stroke="{col}" stroke-width="1.6"/>'
                     + text(bx + 62, 133, name, 13, col, SANS, "middle", "600"))
            b.append(f'<path d="M{bx + 104} 128h20V{y - 13}" fill="none" stroke="{col}" stroke-width="1.6"/>')
            b.append(f'<circle cx="{bx + 124}" cy="{y}" r="12" fill="{BG}" stroke="{col}" stroke-width="2"/>' + text(bx + 124, y + 6, "+", 18, col, MONO, "middle", "700"))
            b.append(f'<circle cx="{bx + 124}" cy="{y}" r="12" fill="{col}" opacity="0">'
                     f'<animate attributeName="opacity" dur="10s" repeatCount="indefinite" values="0;0;0.9;0;0" keyTimes="0;{t0:.2f};{t0 + 0.04:.2f};{t0 + 0.14:.2f};1"/></circle>')
        x += 330
    b.append(f'<rect x="{x - 8}" y="{y - 22}" width="92" height="44" rx="6" fill="{PANEL}" stroke="{INK}"/>' + text(x + 38, y - 1, "norm +", 12, INK, SANS, "middle") + text(x + 38, y + 14, "unembed", 12, INK, SANS, "middle"))
    b.append(f'<circle r="7" fill="{AMBER}"><animateMotion dur="10s" repeatCount="indefinite" path="M70 {y}H890" keyPoints="0;0;1;1" keyTimes="0;0.04;0.9;1" calcMode="linear"/></circle>')
    norms = tr["transformer"]["stream_norms"]
    parts = "   ".join(f"after {'embed' if i == 0 else 'block ' + str(i)}: {n[-1]:.1f}" for i, n in enumerate(norms))
    b.append(text(70, y + 70, "size of the last token's vector:   " + parts, 13.5, INK, MONO))
    b.append(text(70, y + 96, f"each block only ADDS to the stream, never overwrites. {tr['n_parameters']:,} parameters in total.", 13.5, MUTED, SANS))
    return frame(y + 120, "".join(b), "stage 4  /  transformer")


def s05_attention(tr):
    b = []
    pieces, att = tr["tokens"]["pieces"], tr["attention"]
    T = len(pieces)
    xs, x = [], 40.0
    size = 13 if T <= 14 else 12
    for p in pieces:
        w = chip_w(p, size)
        xs.append((x, w))
        x += w + 5
    scale = min(1.0, (W - 80) / (x - 45))
    y = 250
    g = []
    for i, (p, (cx, w)) in enumerate(zip(pieces, xs)):
        c, _ = chip(cx, y, p, CHIPS[i % len(CHIPS)] if i < T - 1 else INK, size)
        g.append(c)
    heads = [(li, h) for li in range(len(att)) for h in range(len(att[li]))]
    n = len(heads)
    lx, lw = xs[-1]
    for k, (li, h) in enumerate(heads):
        t0, t1 = k / n, (k + 1) / n
        arcs = []
        for j in range(T):
            wgt = att[li][h][-1][j]
            if wgt < 0.04:
                continue
            jx, jw = xs[j]
            a, c = jx + jw / 2, lx + lw / 2
            if j == T - 1:
                arcs.append(f'<circle cx="{c}" cy="{y - 40}" r="14" fill="none" stroke="{AMBER}" stroke-width="{1 + wgt * 9:.1f}" opacity="0.85"/>')
            else:
                hgt = min(165, 40 + (c - a) * 0.28)
                arcs.append(f'<path d="M{c:.0f} {y - 22}C{c:.0f} {y - 22 - hgt:.0f} {a:.0f} {y - 22 - hgt:.0f} {a:.0f} {y - 22}" fill="none" '
                            f'stroke="{AMBER}" stroke-width="{1 + wgt * 9:.1f}" opacity="0.85" stroke-linecap="round"/>')
                arcs.append(text(a, y + 28, f"{wgt:.0%}", 12, AMBER, MONO, "middle"))
        label = text(40 / scale, 64 / scale, f"block {li + 1}, head {h + 1}", 15 / scale, INK, SANS, weight="600")
        g.append(f'<g opacity="0"><animate attributeName="opacity" dur="{n * 2.5}s" repeatCount="indefinite" values="0;1;1;0;0" '
                 f'keyTimes="0;{0.02:.3f};{1 / n - 0.02:.3f};{1 / n:.3f};1" begin="{k * 2.5}s"/>{label}{"".join(arcs)}</g>')
    b.append(f'<g transform="scale({scale:.3f})">{"".join(g)}</g>')
    yb = y * scale + 62
    b.append(text(40, yb, "The last token is about to predict the first word of the answer. Line thickness = how much it reads from each earlier token.", 13, MUTED, SANS))
    return frame(int(yb + 24), "".join(b), "stage 5  /  attention close-up   (real attention weights, cycling through every head)")


def curve(xs, ys, x0, y0, w, h, ymax, colour, dur=8, width=2.4):
    pts = " ".join(f"{x0 + w * (x - xs[0]) / (xs[-1] - xs[0]):.1f},{y0 + h - h * min(y, ymax) / ymax:.1f}" for x, y in zip(xs, ys))
    return (f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linejoin="round" '
            f'pathLength="100" stroke-dasharray="100" stroke-dashoffset="100">'
            f'<animate attributeName="stroke-dashoffset" dur="{dur}s" repeatCount="indefinite" values="100;0;0" keyTimes="0;0.7;1"/></polyline>')


def axes(x0, y0, w, h, ymax, ylabel, xlabel, ticks):
    b = [f'<path d="M{x0} {y0}V{y0 + h}H{x0 + w}" fill="none" stroke="{MUTED}" stroke-width="1.2"/>']
    for t in ticks:
        yy = y0 + h - h * t / ymax
        b.append(f'<line x1="{x0}" y1="{yy}" x2="{x0 + w}" y2="{yy}" stroke="{GRID}"/>' + text(x0 - 8, yy + 4, f"{t:g}", 11.5, MUTED, MONO, "end"))
    b.append(text(x0, y0 - 10, ylabel, 12.5, MUTED, SANS) + text(x0 + w, y0 + h + 22, xlabel, 12.5, MUTED, SANS, "end"))
    return "".join(b)


def s06_pretrain(tr):
    b, log = [], tr["pretrain_log"]
    x0, y0, w, h, ymax = 70, 78, 380, 210, 6.5
    b.append(axes(x0, y0, w, h, ymax, "loss  (how surprised the model is by the real next token)", "training step", [0, 2, 4, 6]))
    import math
    guess = math.log(tr["config"]["vocab_size"])
    x0, y0, w, h, ymax = 70, 78, 380, 210, 7.0
    yy = y0 + h - h * guess / ymax
    b.append(f'<line x1="{x0}" y1="{yy}" x2="{x0 + w}" y2="{yy}" stroke="{CORAL}" stroke-dasharray="4 4"/>' + text(x0 + w - 4, yy + 16, f"pure guessing = {guess:.2f}", 11.5, CORAL, SANS, "end"))
    b.append(curve(log["steps"], log["train_loss"], x0, y0, w, h, ymax, AMBER))
    b.append(text(490, 88, "what the model writes after ...", 13, MUTED, SANS))
    total = log["steps"][-1] + 1
    for k, s in enumerate(log["samples"]):
        t0 = 0.02 + 0.7 * (min(s["step"], total) / total) ** 0.5
        yk = 120 + k * 46
        cont = s["text"][len("The weather in Los Angeles is"):].replace("\n", "↵")[:41]
        b.append(f'<g opacity="0">{appear(t0, 0.97, 8)}' + text(490, yk, f"step {s['step']}", 12.5, AMBER, MONO, weight="600")
                 + text(490, yk + 19, "…Los Angeles is", 12.5, MUTED, MONO) + text(490 + 15 * 12.5 * 0.602, yk + 19, cont, 12.5, INK, MONO) + "</g>")
    return frame(340, "".join(b), "stage 6  /  pretraining   (the real loss curve of this repository's model)")


def s07_backprop(tr):
    b, y = [], 150
    names = ["embed", "block 1", "block 2", "logits", "loss"]
    xs = [70 + i * 190 for i in range(5)]
    for i, (n, x) in enumerate(zip(names, xs)):
        last = i == 4
        b.append(f'<rect x="{x}" y="{y - 30}" width="120" height="60" rx="8" fill="{PANEL}" stroke="{CORAL if last else INK}" stroke-width="1.6"/>'
                 + text(x + 60, y + 5, n, 15, CORAL if last else INK, SANS, "middle", "600"))
        if not last:
            b.append(f'<rect x="{x}" y="{y - 30}" width="120" height="60" rx="8" fill="{MINT}" opacity="0">'
                     f'<animate attributeName="opacity" dur="9s" repeatCount="indefinite" values="0;0;0.55;0;0" keyTimes="0;0.84;0.88;0.96;1"/></rect>')
    for i in range(4):
        t0 = 0.04 + 0.09 * i
        b.append(f'<g opacity="0.25">' + arrow(xs[i] + 124, y - 12, xs[i + 1] - 4, y - 12, AMBER, 2.2)
                 + f'<animate attributeName="opacity" dur="9s" repeatCount="indefinite" values="0.25;0.25;1;1;0.25;0.25" keyTimes="0;{t0:.2f};{t0 + 0.03:.2f};0.42;0.46;1"/></g>')
        t1 = 0.5 + 0.08 * (3 - i)
        b.append(f'<g opacity="0.25">' + arrow(xs[i + 1] - 4, y + 12, xs[i] + 124, y + 12, CORAL, 2.2)
                 + f'<animate attributeName="opacity" dur="9s" repeatCount="indefinite" values="0.25;0.25;1;1;0.25;0.25" keyTimes="0;{t1:.2f};{t1 + 0.03:.2f};0.82;0.86;1"/></g>')
    cap = [(0.02, 0.44, AMBER, "1. FORWARD   data flows right. Every layer remembers what it saw."),
           (0.46, 0.82, CORAL, "2. BACKWARD  blame flows left. Chain rule: each layer passes back dLoss/dInput."),
           (0.84, 0.98, MINT, "3. UPDATE    every weight moves a tiny step against its gradient.")]
    for t0, t1, col, s in cap:
        b.append(f'<g opacity="0">{appear(t0, t1, 9)}' + text(70, 244, s, 15, col, MONO, weight="600") + "</g>")
    b.append(text(70, 84, "forward", 12.5, AMBER, SANS) + text(70, y + 60, "backward (gradients)", 12.5, CORAL, SANS))
    b.append(text(70, 282, "w  =  w  -  learning_rate x dLoss/dw        repeated for all "
             f"{tr['n_parameters']:,} weights, {tr['pretrain_log']['steps'][-1] + 1:,} times", 13.5, MUTED, MONO))
    return frame(310, "".join(b), "stage 7  /  backpropagation")


def s08_align(tr):
    b = []
    cols = [("base model", "pretraining only", tr["base_output"], CORAL, "continues the text. Does not answer."),
            ("+ SFT", "Supervised Fine-Tuning", tr["sft_log"]["after"], AMBER, f"answers, but honest only {tr['sft_log']['honesty_rate']:.0%} of the time"),
            ("+ DPO", "Direct Preference Optimization", tr["dpo_log"]["after"], MINT, f"honest in {tr['dpo_log']['honesty_after']:.0%} of samples")]
    b.append(text(40, 72, "same prompt to all three:", 13, MUTED, SANS) + text(222, 72, tr["prompt"], 14, INK, MONO, weight="600"))
    for k, (name, sub, out, col, note) in enumerate(cols):
        x, t0 = 40 + k * 300, 0.05 + 0.28 * k
        b.append(f'<rect x="{x}" y="92" width="280" height="196" rx="10" fill="{PANEL}" stroke="{col}" stroke-width="1.6"/>')
        b.append(text(x + 16, 120, name, 17, col, SANS, weight="700") + text(x + 16, 140, sub, 12, MUTED, SANS))
        words, lines, cur = out.replace("\n", " ↵ ").split(" "), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 > 31:
                lines.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines.append(cur)
        g = "".join(text(x + 16, 170 + i * 19, ln, 13, INK, MONO) for i, ln in enumerate(lines[:5]))
        b.append(f'<g opacity="0">{appear(t0)}{g}' + text(x + 16, 274, note, 11.5, col, SANS) + "</g>")
        if k < 2:
            b.append(arrow(x + 282, 190, x + 298, 190, MUTED, 2))
    return frame(312, "".join(b), "stage 8  /  alignment   (real outputs from the three saved checkpoints)")


def s09_sample(tr):
    b, demo = [], tr["sampling_demo"]
    frames = [("SFT checkpoint (before DPO): the model is torn, so the dice decide the whole answer", demo["sft"], AMBER,
               {" R": "-> 'Right now it is 75 degrees...'", " I": "-> 'I cannot see live weather data...'"}),
              ("aligned checkpoint (after DPO): same prompt, the weights have made up their mind", demo["aligned"], MINT, {})]
    n = len(frames)
    for k, (title, cands, col, notes) in enumerate(frames):
        g = [text(60, 80, "first token of the answer.  " + title, 13.5, col, SANS, weight="600")]
        for i, c in enumerate(cands):
            y = 122 + i * 38
            wbar = max(3, c["p"] * 430)
            g.append(text(170, y + 5, show(c["piece"]), 15, INK, MONO, "end", "600"))
            g.append(f'<rect x="186" y="{y - 13}" width="{wbar:.0f}" height="24" rx="4" fill="{col if i == 0 else CYAN}" opacity="{1 if i == 0 else 0.6}"/>')
            g.append(text(196 + wbar, y + 5, f"{c['p']:.1%}   " + notes.get(c["piece"], ""), 13, MUTED if i else col, MONO))
        b.append(f'<g opacity="0"><animate attributeName="opacity" dur="{n * 4.5}s" begin="{k * 4.5}s" repeatCount="indefinite" '
                 f'values="0;1;1;0;0" keyTimes="0;0.02;{1 / n - 0.02:.3f};{1 / n:.3f};1"/>{"".join(g)}</g>')
    b.append(text(60, 326, "The model outputs a probability for EVERY token. Sampling rolls a weighted die. Then the whole model runs again.", 13, MUTED, SANS))
    return frame(350, "".join(b), "stage 9  /  sampling   (real probabilities from two saved checkpoints, temperature 1.0)")


def s10_output(tr):
    b = []
    gens = [g for g in tr["generation"] if g["piece"] != "<|end|>"]
    x, y = 60.0, 120
    for i, g in enumerate(gens):
        w = chip_w(g["piece"], 15)
        if x + w > W - 50:
            x, y = 60.0, y + 66
        t0 = 0.03 + 0.6 * i / len(gens)
        c, _ = chip(x, y, g["piece"], CHIPS[i % len(CHIPS)], 15)
        b.append(f'<g opacity="0">{appear(t0)}{c}' + text(x + w / 2, y + 24, str(g["id"]), 11.5, CYAN, MONO, "middle") + "</g>")
        x += w + 6
    b.append(f'<g opacity="0">{appear(0.68)}' + text(60, y + 84, "decode: look up each id's bytes, join them, read as text", 13, MUTED, SANS)
             + text(60, y + 120, tr["output"], 19, MINT, MONO, weight="700") + "</g>")
    return frame(y + 150, "".join(b), "stage 10  /  output   (generated one token at a time, then decoded)")


def lens(tr):
    """Logit lens: the prediction you would read off the residual stream after each layer, animated row by row."""
    b, rows = [], tr["logit_lens"]
    names = ["after embedding"] + [f"after block {i}" for i in range(1, len(rows))]
    b.append(text(60, 78, "if the model stopped HERE and read a prediction off the stream, what would it say?", 13, MUTED, SANS))
    for i, (name, row) in enumerate(zip(names, rows)):
        y, t0 = 122 + i * 62, 0.05 + 0.28 * i
        g = [text(60, y + 5, name, 15, INK, SANS, weight="600")]
        x = 260
        for c in row[:3]:
            w = max(4, c["p"] * 260)
            g.append(f'<rect x="{x}" y="{y - 12}" width="{w:.0f}" height="24" rx="4" fill="{CYAN if c is not row[0] else AMBER}" opacity="0.9"/>')
            g.append(text(x + w + 8, y + 5, f"{show(c['piece'])} {c['p']:.0%}", 13, INK, MONO))
            x += w + 8 + (len(show(c["piece"])) + 5) * 13 * 0.6 + 16
        b.append(f'<g opacity="0">{appear(t0)}{"".join(g)}</g>')
        if i:
            b.append(f'<g opacity="0">{appear(t0 - 0.03)}' + arrow(150, y - 46, 150, y - 18, MUTED, 1.6) + "</g>")
    yb = 122 + len(rows) * 62
    b.append(text(60, yb + 6, "The embedding alone is clueless. Each block adds to the stream until the answer is obvious.", 13.5, MUTED, SANS))
    return frame(yb + 30, "".join(b), "stage 4  /  the logit lens   (real numbers: what each layer 'thinks' the next token is)")


def attention_grid(tr):
    """All heads as small multiples: the standard picture in interpretability papers."""
    pieces, att = tr["tokens"]["pieces"], tr["attention"]
    T, L, H = len(pieces), len(att), len(att[0])
    cell, gap = 12, 10
    size = T * cell
    b = []
    for l in range(L):
        for h in range(H):
            x0, y0 = 60 + h * (size + 70), 96 + l * (size + 56)
            t0 = 0.04 + 0.1 * (l * H + h)
            g = [f'<rect x="{x0 - 3}" y="{y0 - 3}" width="{size + 6}" height="{size + 6}" rx="4" fill="{PANEL}" stroke="{GRID}"/>']
            for i in range(T):
                for j in range(i + 1):
                    v = att[l][h][i][j]
                    if v > 0.02:
                        g.append(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell - 1}" height="{cell - 1}" fill="{AMBER}" opacity="{min(1, v * 1.3):.2f}"/>')
            g.append(text(x0 + size / 2, y0 + size + 18, f"block {l + 1} head {h + 1}", 12, MUTED, SANS, "middle"))
            b.append(f'<g opacity="0">{appear(t0, 0.97, 12)}{"".join(g)}</g>')
    yb = 96 + L * (size + 56)
    b.append(text(60, 78, "rows: the token doing the looking (top = first token). columns: the token being looked at. brighter = more attention.", 12.5, MUTED, SANS))
    b.append(text(60, yb + 4, "Every matrix is a triangle: the causal mask. A bright diagonal is 'read myself'; a bright column is one token everybody reads.", 13, MUTED, SANS))
    return frame(yb + 28, "".join(b), f"stage 5  /  all {L * H} attention heads at once   (real weights for this prompt)")


def main() -> None:
    tr = json.loads((ROOT / "artifacts" / "trace.json").read_text())
    ASSETS.mkdir(exist_ok=True)
    out = {"hero": hero(tr), "lens": lens(tr), "attention_grid": attention_grid(tr)}
    for i in range(1, 11):
        out[f"pipeline_{i:02d}"] = pipeline(i)
    for i, fn in enumerate([s01_input, s02_tokens, s03_embed, s04_transformer, s05_attention,
                            s06_pretrain, s07_backprop, s08_align, s09_sample, s10_output], start=1):
        out[f"stage_{i:02d}"] = fn(tr)
    for name, svg in out.items():
        (ASSETS / f"{name}.svg").write_text(svg)
    print(f"wrote {len(out)} animated SVG files -> assets/")


if __name__ == "__main__":
    main()
