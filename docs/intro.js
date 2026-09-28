/*
 * intro.js - a 32-second film of the real model answering one prompt, looping forever.
 * Every number drawn comes from the model running in your browser (engine.js), not from a recording.
 *
 *   Intro.mount(container, { engine, prompt })   ->  { seek(t), play(), pause(), setPrompt(p), duration }
 *
 * The picture is a pure function of time t, so the same code plays live on the website and is recorded
 * frame by frame into assets/intro.gif for the README (tools/record_intro.py).
 */
(function (root) {
  "use strict";
  const W = 1280, H = 720;
  const BG = "#0F2A47", GRID = "#1B3F66", PANEL = "#143556", INK = "#DCE9F5", MUTED = "#7FA3C7";
  const AMBER = "#FFB238", CORAL = "#FF6F61", MINT = "#6FE3B4", CYAN = "#5CC8FF";
  const CHIP = ["#FFB238", "#5CC8FF", "#6FE3B4", "#C9A7FF", "#FF6F61", "#9AD0FF"];
  const MONO = "'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace", SANS = "'IBM Plex Sans', system-ui, sans-serif";

  // [start, end, rail label, caption, lesson anchor]
  const SCENES = [
    [0, 3.5, "Input", "You type a question. To the computer it is only a row of bytes, one per character.", "#1"],
    [3.5, 7, "Tokens", "The tokenizer cuts the text into tokens and gives each an id. Special tokens mark who is speaking.", "#2"],
    [7, 10, "Embedding", "Each id looks up a row of 64 learned numbers: its vector. Blue is positive, coral negative.", "#3"],
    [10, 17, "Transformer", "Two blocks. Attention lets the last token read the earlier ones; 256 perceptrons then process it. Each block ADDS to the stream.", "#4"],
    [17, 20, "Scores", "The last vector is compared with all 768 token vectors. Softmax turns the scores into probabilities.", "#9"],
    [20, 26, "Sample, repeat", "The winning token is appended and the whole network runs again, once per token, until it predicts <|end|>.", "#10"],
    [26, 29, "Output", "The chosen ids are decoded back into text. That is the entire trick.", "#10"],
    [29, 32.5, "Harness", "In a real assistant, software wraps the model: guards, memory, a tool call for live data, and a check on the answer.", "harness.html"],
  ];
  const DURATION = 32.5;

  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const prog = (t, a, b) => clamp((t - a) / (b - a));
  const ease = x => (x = clamp(x), x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const lerp = (a, b, u) => a + (b - a) * u;
  const show = s => s.replace(/ /g, "·").replace(/\n/g, "↵");
  function heat(v, m) {
    const t = clamp(v / m, -1, 1), a = Math.pow(Math.abs(t), 0.7), b = [20, 53, 86], c = t >= 0 ? [92, 200, 255] : [255, 111, 97];
    return `rgb(${b.map((x, k) => Math.round(x + (c[k] - x) * a)).join(",")})`;
  }

  // -------------------------------------------------------------- data from the real model
  function prepare(E, prompt) {
    const ids = E.encode(E.formatPrompt(prompt)).slice(-(E.config.context_length - 20));
    const R = E.inspect("aligned", ids), T = ids.length, D = E.D, last = T - 1;
    const endId = E.encode("<|end|>")[0], gen = [];
    let cur = ids.slice();
    for (let n = 0; n < 40; n++) {
      const p = E.softmax(Array.from(E.logitsFor("aligned", cur)));
      const order = p.map((_, i) => i).sort((a, b) => p[b] - p[a]).slice(0, 6);
      gen.push({ id: order[0], piece: E.piece(order[0]), top: order.map(i => ({ piece: E.piece(i), p: p[i] })) });
      if (order[0] === endId) break;
      cur.push(order[0]);
    }
    const attn = R.cap.attention.map(maps => {                       // average over heads, last row
      const row = new Array(T).fill(0);
      maps.forEach(m => m[last].forEach((v, j) => (row[j] += v / maps.length)));
      return row;
    });
    const dims = 24, grab = arr => Array.from({ length: T }, (_, t) => Array.from(arr.subarray(t * D, t * D + dims)));
    const streams = R.cap.stream.map(grab);
    const mx = Math.max(...streams.flat(2).map(Math.abs)) * 0.6;
    const mlp = R.cap.mlpHidden.map(h => Array.from(h.subarray(last * E.DFF, (last + 1) * E.DFF)));
    const answer = E.decode(gen.filter(g => g.id !== endId).map(g => g.id)).trim();
    return { prompt, ids, pieces: ids.map(E.piece), T, attn, streams, mx, mlp, gen, answer, endId, nParams: 153344 };
  }

  // -------------------------------------------------------------- drawing one frame at time t
  function frame(ctx, d, t) {
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#1B3F6655"; ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    const fadeAll = 1 - prog(t, DURATION - 0.6, DURATION);            // fade to restart
    ctx.globalAlpha = fadeAll;

    // ---- title + scene rail
    ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
    ctx.font = `700 22px ${SANS}`; ctx.fillStyle = INK; ctx.fillText("transparent-transformer-llm", 40, 44);
    ctx.font = `14px ${SANS}`; ctx.fillStyle = MUTED; ctx.fillText("what really happens when you press Enter  ·  live numbers from the model", 40, 66);
    const scene = SCENES.findIndex(s => t < s[1]), rx0 = 690, rw = 550;
    SCENES.forEach((s, i) => {
      const x = rx0 + (rw * i) / (SCENES.length - 1), now = i === scene, done = i < scene;
      if (i) { ctx.strokeStyle = done || now ? MINT : GRID; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x - rw / (SCENES.length - 1) + 9, 44); ctx.lineTo(x - 9, 44); ctx.stroke(); }
      ctx.beginPath(); ctx.arc(x, 44, now ? 9 : 6, 0, 7); ctx.fillStyle = now ? AMBER : done ? MINT : GRID; ctx.fill();
      ctx.font = `${now ? 600 : 400} 11px ${SANS}`; ctx.fillStyle = now ? INK : MUTED; ctx.textAlign = "center"; ctx.fillText(s[2], x, 68); ctx.textAlign = "left";
    });

    const fs = Math.min(17, 820 / Math.max(40, d.pieces.join("").length + d.T * 1.2));
    ctx.font = `600 ${fs}px ${MONO}`;
    const cw = ctx.measureText("M").width;
    // chip geometry for the prompt tokens
    let x = 40; const chips = d.pieces.map(p => { const w = show(p).length * cw + 12; const c = { x, w, cx: x + w / 2 }; x += w + 6; return c; });
    const rowY = 150;

    // ---- 1 input: typewriter, then 2 tokens: text splits into chips
    const typed = Math.floor(ease(prog(t, 0.3, 2.6)) * d.prompt.length);
    const split = ease(prog(t, 3.6, 5.2));
    if (t < 3.6) {
      ctx.font = `600 24px ${MONO}`; ctx.fillStyle = AMBER;
      ctx.fillText(d.prompt.slice(0, typed) + (Math.floor(t * 3) % 2 && typed < d.prompt.length ? "▍" : ""), 40, rowY);
      if (t > 1.2) {                                                   // bytes trickle underneath
        ctx.font = `12px ${MONO}`; ctx.fillStyle = CYAN;
        const bytes = Array.from(new TextEncoder().encode(d.prompt.slice(0, typed))).join(" ");
        ctx.globalAlpha = fadeAll * prog(t, 1.2, 1.8); ctx.fillText(bytes.slice(0, 150), 40, rowY + 34); ctx.globalAlpha = fadeAll;
      }
    } else {
      ctx.font = `600 ${fs}px ${MONO}`; ctx.textBaseline = "middle";
      d.pieces.forEach((p, i) => {
        const c = chips[i], y = rowY + (1 - split) * (i % 2 ? -8 : 8), sp = special(p);
        ctx.globalAlpha = fadeAll * (sp ? split : 1);
        ctx.fillStyle = sp ? INK : CHIP[i % 6]; round(ctx, c.x, y - fs * 0.75, c.w, fs * 1.5, 5); ctx.fill();
        ctx.fillStyle = "#0B1E33"; ctx.fillText(show(p), c.x + 6, y + 1);
        ctx.globalAlpha = fadeAll * prog(t, 5, 6); ctx.font = `11px ${MONO}`; ctx.fillStyle = CYAN; ctx.textAlign = "center";
        ctx.fillText(String(d.ids[i]), c.cx, y + fs + 4); ctx.textAlign = "left"; ctx.font = `600 ${fs}px ${MONO}`;
      });
      ctx.globalAlpha = fadeAll; ctx.textBaseline = "alphabetic";
    }

    // ---- 3 embedding strips, which change colour as the blocks add to them
    const stripTop = 196, cell = 6.2, dims = d.streams[0][0].length;
    if (t >= 7) {
      const grow = prog(t, 7, 9.2);
      const b1 = prog(t, 11.9, 12.6), b2 = prog(t, 15.2, 15.9);        // stream after block 1 and 2
      d.pieces.forEach((_, i) => {
        const c = chips[i], shown = Math.floor(dims * clamp(grow * 1.2 - (i / d.T) * 0.2));
        for (let k = 0; k < shown; k++) {
          const v = lerp(lerp(d.streams[0][i][k], d.streams[1][i][k], b1), d.streams[2][i][k], b2);
          ctx.fillStyle = heat(v, d.mx); ctx.fillRect(c.cx - 7, stripTop + k * cell, 14, cell - 1);
        }
      });
      ctx.globalAlpha = fadeAll * prog(t, 8, 9) * (1 - prog(t, 9.6, 10));
      ctx.font = `12px ${SANS}`; ctx.fillStyle = MUTED; ctx.fillText(`${d.T} tokens × 64 numbers (first ${dims} shown)`, 40, stripTop + dims * cell + 16);
      ctx.globalAlpha = fadeAll;
    }

    // ---- 4 transformer: two bands, attention arcs, residual packet, MLP neuron grids
    const bands = [[372, "block 1", 10, 13.4], [470, "block 2", 13.4, 17]], lastC = chips[d.T - 1], right = chips[d.T - 1].x + chips[d.T - 1].w;
    if (t >= 10) {
      const lineTop = stripTop + dims * cell;
      ctx.strokeStyle = "#FFB23866"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(lastC.cx, lineTop); ctx.lineTo(lastC.cx, lerp(lineTop, 548, ease(prog(t, 10, 16.8)))); ctx.stroke();
      bands.forEach(([y, label, a, b], l) => {
        const on = prog(t, a, a + 0.5); if (!on) return;
        ctx.globalAlpha = fadeAll * on;
        ctx.strokeStyle = "#7FA3C788"; ctx.setLineDash([5, 5]); round(ctx, 30, y - 44, right - 20, 80, 10); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = `600 12px ${SANS}`; ctx.fillStyle = MUTED; ctx.fillText(label, 40, y - 28);
        // attention arcs from the last token to every earlier one (average of the 4 heads)
        const draw = ease(prog(t, a + 0.3, a + 2));
        d.attn[l].forEach((w, j) => {
          if (j === d.T - 1 || w < 0.03) return;
          const c = chips[j], h = Math.min(60, 18 + (lastC.cx - c.cx) * 0.09);
          ctx.strokeStyle = `rgba(255,178,56,${(0.25 + w).toFixed(2)})`; ctx.lineWidth = 1 + w * 12;
          ctx.beginPath();
          const steps = 24, n = Math.floor(steps * draw);
          for (let s = 0; s <= n; s++) {
            const u = s / steps, px = lerp(lastC.cx, c.cx, u), py = y + 22 - Math.sin(u * Math.PI) * h;
            s ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
          }
          ctx.stroke();
          if (draw > 0.95 && w > 0.12) { ctx.font = `11px ${MONO}`; ctx.fillStyle = AMBER; ctx.textAlign = "center"; ctx.fillText(`${Math.round(w * 100)}%`, c.cx, y + 34); ctx.textAlign = "left"; }
        });
        // the + of the residual connection
        const add = prog(t, b - 1.2, b - 0.6);
        ctx.beginPath(); ctx.arc(lastC.cx, y + 22, 10 + 6 * Math.sin(add * Math.PI), 0, 7); ctx.strokeStyle = add > 0 && add < 1 ? AMBER : MUTED; ctx.lineWidth = 2; ctx.stroke();
        ctx.font = `700 15px ${MONO}`; ctx.fillStyle = INK; ctx.textAlign = "center"; ctx.fillText("+", lastC.cx, y + 27); ctx.textAlign = "left";
        // 256 perceptrons of the last token, lighting up in a sweep
        const gx = right + 40, gy = y - 34, m = Math.max(...d.mlp[l]) || 1, lit = prog(t, a + 1.6, b - 1);
        ctx.font = `11px ${SANS}`; ctx.fillStyle = MUTED; ctx.fillText("MLP · 256 perceptrons", gx, gy - 6);
        d.mlp[l].forEach((v, k) => {
          const on = (k * 97 % 256) / 256 < lit, px = gx + (k % 32) * 6.5, py = gy + Math.floor(k / 32) * 7.5;
          ctx.fillStyle = on && v > 0.03 * m ? `rgba(111,227,180,${clamp(0.25 + v / m).toFixed(2)})` : GRID;
          ctx.fillRect(px, py, 5, 6);
        });
        if (lit >= 1) { const act = d.mlp[l].filter(v => v > 0.05).length; ctx.fillStyle = MINT; ctx.fillText(`${act} of 256 firing`, gx + 132, gy - 6); }
        ctx.globalAlpha = fadeAll;
      });
      // the packet travelling down the residual stream
      const py = lerp(lineTop, 548, ease(prog(t, 10, 16.8)));
      if (t < 17.5) { ctx.beginPath(); ctx.arc(lastC.cx, py, 7 + Math.sin(t * 8) * 1.5, 0, 7); ctx.fillStyle = AMBER; ctx.fill(); }
    }

    // ---- 5 scores -> probabilities, updated at every generated token in scene 6
    const panelX = 1000, panelY = 150;
    const nGen = d.gen.length, sampleStart = 20.8, per = (26 - sampleStart) / Math.max(1, nGen - 1);
    const step = t < sampleStart ? 0 : Math.min(nGen - 1, Math.floor((t - sampleStart) / per) + 1);
    if (t >= 17) {
      const appear = ease(prog(t, 17, 18.6)), g = d.gen[Math.min(step, nGen - 1)];
      ctx.globalAlpha = fadeAll * prog(t, 17, 17.6) * (1 - prog(t, 26.5, 27.2));
      ctx.fillStyle = PANEL; round(ctx, panelX - 16, panelY - 34, 270, 300, 12); ctx.fill();
      ctx.font = `600 13px ${SANS}`; ctx.fillStyle = INK; ctx.fillText("next-token probabilities", panelX, panelY - 12);
      g.top.forEach((c, i) => {
        const y = panelY + 16 + i * 42, w = Math.max(3, c.p * 110 * appear);
        ctx.fillStyle = i === 0 ? MINT : CYAN; ctx.globalAlpha *= i === 0 ? 1 : 0.7; round(ctx, panelX + 88, y - 11, w, 20, 4); ctx.fill(); ctx.globalAlpha = fadeAll * prog(t, 17, 17.6) * (1 - prog(t, 26.5, 27.2));
        ctx.font = `600 14px ${MONO}`; ctx.fillStyle = INK; ctx.textAlign = "right"; ctx.fillText(show(c.piece).slice(0, 10), panelX + 80, y + 4); ctx.textAlign = "left";
        ctx.font = `12px ${MONO}`; ctx.fillStyle = MUTED; ctx.fillText(`${(c.p * 100).toFixed(1)}%`, panelX + 94 + w, y + 4);
      });
      if (t >= sampleStart - 0.6) { ctx.font = `12px ${MONO}`; ctx.fillStyle = AMBER; ctx.fillText(`forward pass ${step + 1} of ${nGen}`, panelX, panelY + 262); }
      ctx.globalAlpha = fadeAll;
    }

    // ---- 6 sampling: chosen tokens fly to the answer line; 7 decoded text
    const outY = 604;
    if (t >= 19.2) {
      ctx.font = `12px ${SANS}`; ctx.fillStyle = MUTED; ctx.fillText("answer", 40, outY - 22);
      ctx.font = `600 15px ${MONO}`;
      const ow = ctx.measureText("M").width; let ox = 40;
      const visible = d.gen.filter(g => g.id !== d.endId), merge = ease(prog(t, 26.3, 27.4));
      visible.forEach((g, i) => {
        const tIn = i === 0 ? 19.4 : sampleStart + i * per, a = ease(prog(t, tIn, tIn + 0.45)); const w = show(g.piece).length * ow + 10;
        if (a > 0) {
          const sx = lerp(panelX + 88, ox, a), sy = lerp(panelY + 16, outY, a);
          ctx.globalAlpha = fadeAll * (1 - merge);
          ctx.fillStyle = CHIP[i % 6]; round(ctx, sx, sy - 12, w, 24, 5); ctx.fill();
          ctx.fillStyle = "#0B1E33"; ctx.textBaseline = "middle"; ctx.fillText(show(g.piece), sx + 5, sy + 1); ctx.textBaseline = "alphabetic";
        }
        ox += w + 4;
      });
      ctx.globalAlpha = fadeAll * merge; ctx.font = `600 22px ${MONO}`; ctx.fillStyle = MINT; ctx.fillText(d.answer, 40, outY + 8);
      ctx.globalAlpha = fadeAll;
    }

    // ---- 8 harness: the ring of software around the model
    if (t >= 29) {
      const a = ease(prog(t, 29, 29.6));
      ctx.globalAlpha = fadeAll * a * 0.93; ctx.fillStyle = BG; ctx.fillRect(0, 84, W, 560); ctx.globalAlpha = fadeAll * a;
      const cx = 640, cy = 350, parts = ["input guard", "memory", "router", "tool: live weather", "prompt builder", "output guard"];
      const spot = i => { const ang = -Math.PI / 2 + (i / parts.length) * Math.PI * 2; return [cx + Math.cos(ang) * 330, cy + Math.sin(ang) * 190]; };
      parts.forEach((_, i) => {                                         // spokes first, so boxes sit on top of them
        const [px, py] = spot(i), lit = prog(t, 29.6 + i * 0.35, 29.9 + i * 0.35);
        ctx.strokeStyle = lit ? MINT : GRID; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();
      });
      ctx.fillStyle = PANEL; ctx.strokeStyle = AMBER; ctx.lineWidth = 2; round(ctx, cx - 110, cy - 40, 220, 80, 12); ctx.fill(); ctx.stroke();
      ctx.font = `700 18px ${SANS}`; ctx.fillStyle = INK; ctx.textAlign = "center"; ctx.fillText("the model", cx, cy - 4);
      ctx.font = `12px ${SANS}`; ctx.fillStyle = MUTED; ctx.fillText("stages 1 to 10, everything above", cx, cy + 18);
      parts.forEach((p, i) => {
        const [px, py] = spot(i), lit = prog(t, 29.6 + i * 0.35, 29.9 + i * 0.35);
        ctx.fillStyle = PANEL; round(ctx, px - 88, py - 22, 176, 44, 10); ctx.fill();          // solid base hides the spoke
        ctx.fillStyle = lit ? "#6FE3B433" : PANEL; ctx.strokeStyle = lit ? MINT : MUTED; round(ctx, px - 88, py - 22, 176, 44, 10); ctx.fill(); ctx.stroke();
        ctx.font = `600 14px ${SANS}`; ctx.fillStyle = lit ? INK : MUTED; ctx.fillText(p, px, py + 5);
      });
      if (t > 31) { ctx.font = `12px ${MONO}`; ctx.fillStyle = CYAN; ctx.fillText('get_weather("Los Angeles")  →  live data pasted into the prompt as text', cx, cy + 240); }
      ctx.textAlign = "left"; ctx.globalAlpha = fadeAll;
    }

    // ---- caption band
    const s = SCENES[Math.max(0, scene)];
    ctx.fillStyle = "#0B1E33"; ctx.fillRect(0, 648, W, 72);
    ctx.fillStyle = AMBER; ctx.fillRect(0, 648, W * (t / DURATION), 3);
    ctx.font = `700 15px ${SANS}`; ctx.fillStyle = AMBER; ctx.fillText(`${Math.max(0, scene) + 1} · ${s[2]}`, 40, 682);
    ctx.font = `15px ${SANS}`; ctx.fillStyle = INK; ctx.fillText(s[3], 240, 682);
    ctx.font = `12px ${SANS}`; ctx.fillStyle = MUTED; ctx.fillText(`prompt: ${d.prompt}   ·   ${d.nParams.toLocaleString()} weights   ·   every number is computed live`, 40, 706);
    ctx.globalAlpha = 1;
  }

  function round(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }
  function special(p) { return p.startsWith("<|") && p.endsWith("|>"); }

  // -------------------------------------------------------------- the player
  function mount(el, opts) {
    const E = opts.engine, canvas = document.createElement("canvas");
    canvas.width = W * 2; canvas.height = H * 2; canvas.style.cssText = "width:100%;height:auto;display:block;border-radius:14px;border:1px solid #1B3F66";
    canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "Looping animation of the model answering a prompt, stage by stage");
    el.appendChild(canvas);
    const ctx = canvas.getContext("2d"); ctx.scale(2, 2);
    let data = prepare(E, opts.prompt || "What is Los Angeles like in summer?"), t = 0, playing = true, last = null;
    const still = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function tick(now) {
      if (last !== null && playing) t = (t + (now - last) / 1000) % DURATION;
      last = now; frame(ctx, data, still ? 27.5 : t); if (!opts.manual) root.requestAnimationFrame(tick);
    }
    if (!opts.manual) root.requestAnimationFrame(tick); else frame(ctx, data, 0);
    return {
      duration: DURATION, scenes: SCENES,
      seek(x) { t = clamp(x, 0, DURATION - 1e-3); frame(ctx, data, t); },
      play() { playing = true; }, pause() { playing = false; }, toggle() { playing = !playing; return playing; },
      setPrompt(p) { data = prepare(E, p); t = 0; },
      time: () => t, canvas,
    };
  }
  root.Intro = { mount, DURATION, SCENES };
})(typeof window !== "undefined" ? window : globalThis);
