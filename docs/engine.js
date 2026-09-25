/*
 * engine.js - the SAME model as the Python package, in about 250 lines of plain JavaScript,
 * so the classroom website can run it live in your browser. No libraries, no server.
 *
 * Python file                          what it becomes here
 * transparent_transformer/tokenizer.py   -> encode(), decode(), mergeSteps()
 * transparent_transformer/embedding.py   -> first lines of forward()
 * transparent_transformer/attention.py   -> attention() inside forward()
 * transparent_transformer/layers.py      -> layerNorm(), linear(), gelu()
 * transparent_transformer/sampling.py    -> sampleNext(), generate()
 * transparent_transformer/trace.py       -> buildTrace()
 */
(function (root) {
  "use strict";
  function createEngine(M) {
  const cfg = M.config;
  const D = cfg.d_model, H = cfg.n_heads, HD = D / H, V = cfg.vocab_size, CTX = cfg.context_length;

  // ------------------------------------------------------------------ weights
  function halfToFloat(h) {                  // 16-bit floats halve the download; JavaScript has no native type for them
    const out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) {
      const s = (h[i] & 0x8000) ? -1 : 1, e = (h[i] >> 10) & 0x1f, f = h[i] & 0x3ff;
      out[i] = s * (e === 0 ? (f / 1024) * 2 ** -14 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15));
    }
    return out;
  }
  const cache = {};
  function weights(name) {
    if (cache[name]) return cache[name];
    const out = {};
    for (const [k, v] of Object.entries(M.models[name])) {
      const bin = atob(v.data), bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      out[k] = v.dtype === "f2" ? halfToFloat(new Uint16Array(bytes.buffer)) : new Float32Array(bytes.buffer);
    }
    return (cache[name] = out);
  }

  // ---------------------------------------------------------------- tokenizer
  const vocab = [];
  for (let i = 0; i < 256; i++) vocab.push([i]);
  const ranks = new Map();
  M.merges.forEach(([a, b], i) => { vocab.push(vocab[a].concat(vocab[b])); ranks.set(a * 100000 + b, i); });
  const special = M.special, specialById = {};
  for (const [s, id] of Object.entries(special)) { specialById[id] = s; vocab[id] = Array.from(new TextEncoder().encode(s)); }
  const WORD = / ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
  const SPECIAL_RE = new RegExp("(" + Object.keys(special).map(s => s.replace(/[|]/g, "\\|")).join("|") + ")");
  const dec = new TextDecoder("utf-8");

  function mergeSteps(word, maxRank = Infinity) {   // every intermediate state while one word is merged; maxRank = "only the first k merges exist yet"
    let ids = Array.from(new TextEncoder().encode(word));
    const steps = [ids.slice()];
    while (ids.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = ranks.get(ids[i] * 100000 + ids[i + 1]);
        if (r !== undefined && r < maxRank && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = ids[best], b = ids[best + 1], out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(256 + bestRank); i++; } else out.push(ids[i]);
      }
      ids = out; steps.push(ids.slice());
    }
    return steps;
  }
  function encodeUpTo(text, k) {           // tokenize as the tokenizer would have after only k merges were learned
    const ids = [];
    for (const chunk of text.split(SPECIAL_RE)) {
      if (chunk in special) ids.push(special[chunk]);
      else if (chunk) for (const w of chunk.match(WORD) || []) { const s = mergeSteps(w, k); ids.push(...s[s.length - 1]); }
    }
    return ids;
  }
  function encode(text) {
    const ids = [];
    for (const chunk of text.split(SPECIAL_RE)) {
      if (chunk in special) ids.push(special[chunk]);
      else if (chunk) for (const w of chunk.match(WORD) || []) { const s = mergeSteps(w); ids.push(...s[s.length - 1]); }
    }
    return ids;
  }
  const piece = id => dec.decode(new Uint8Array(vocab[id]));
  const decode = ids => dec.decode(new Uint8Array(ids.flatMap(i => vocab[i])));

  // ------------------------------------------------------------------- layers
  function linear(x, T, W, b, nin, nout) {           // y = x W + b     x:(T,nin) -> (T,nout)
    const y = new Float32Array(T * nout);
    for (let t = 0; t < T; t++) for (let j = 0; j < nout; j++) {
      let s = b[j];
      for (let i = 0; i < nin; i++) s += x[t * nin + i] * W[i * nout + j];
      y[t * nout + j] = s;
    }
    return y;
  }
  function layerNorm(x, T, g, b) {
    const y = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      let mu = 0, va = 0;
      for (let i = 0; i < D; i++) mu += x[t * D + i];
      mu /= D;
      for (let i = 0; i < D; i++) va += (x[t * D + i] - mu) ** 2;
      const sd = Math.sqrt(va / D + 1e-5);
      for (let i = 0; i < D; i++) y[t * D + i] = (x[t * D + i] - mu) / sd * g[i] + b[i];
    }
    return y;
  }
  const A = Math.sqrt(2 / Math.PI);
  const gelu = v => 0.5 * v * (1 + Math.tanh(A * (v + 0.044715 * v * v * v)));
  const rowNorms = (x, T) => Array.from({ length: T }, (_, t) => { let s = 0; for (let i = 0; i < D; i++) s += x[t * D + i] ** 2; return Math.sqrt(s); });

  // ------------------------------------------------------------- forward pass
  function forward(w, ids, capture) {
    const T = ids.length, x = new Float32Array(T * D), cap = { attention: [], attnNorms: [], mlpNorms: [], streamNorms: [] };
    for (let t = 0; t < T; t++) for (let i = 0; i < D; i++)                       // stage 3: embedding
      x[t * D + i] = w["embed.tok"][ids[t] * D + i] + w["embed.pos"][t * D + i];
    if (capture) { cap.embed = Float32Array.from(x); cap.streamNorms.push(rowNorms(x, T)); cap.stream = [Float32Array.from(x)]; }

    for (let l = 0; l < cfg.n_layers; l++) {                                      // stage 4: the blocks
      const p = `block${l}.`;
      // ---- attention (stage 5) ----
      const n1 = layerNorm(x, T, w[p + "ln1.g"], w[p + "ln1.b"]);
      const qkv = linear(n1, T, w[p + "attn.qkv.W"], w[p + "attn.qkv.b"], D, 3 * D);
      const mixed = new Float32Array(T * D), maps = [];
      for (let h = 0; h < H; h++) {
        const map = [];
        for (let i = 0; i < T; i++) {
          const sc = new Float64Array(i + 1); let mx = -Infinity;
          for (let j = 0; j <= i; j++) {                                          // j <= i  IS the causal mask
            let s = 0;
            for (let k = 0; k < HD; k++) s += qkv[i * 3 * D + h * HD + k] * qkv[j * 3 * D + D + h * HD + k];
            sc[j] = s / Math.sqrt(HD); if (sc[j] > mx) mx = sc[j];
          }
          let z = 0; for (let j = 0; j <= i; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
          const row = new Array(T).fill(0);
          for (let j = 0; j <= i; j++) {
            const a = sc[j] / z; row[j] = a;
            for (let k = 0; k < HD; k++) mixed[i * D + h * HD + k] += a * qkv[j * 3 * D + 2 * D + h * HD + k];
          }
          map.push(row);
        }
        maps.push(map);
      }
      const att = linear(mixed, T, w[p + "attn.proj.W"], w[p + "attn.proj.b"], D, D);
      for (let i = 0; i < x.length; i++) x[i] += att[i];                          // residual add
      // ---- multi-layer perceptron ----
      const n2 = layerNorm(x, T, w[p + "ln2.g"], w[p + "ln2.b"]);
      const up = linear(n2, T, w[p + "mlp.up.W"], w[p + "mlp.up.b"], D, cfg.d_ff).map(gelu);
      const mlp = linear(up, T, w[p + "mlp.down.W"], w[p + "mlp.down.b"], cfg.d_ff, D);
      for (let i = 0; i < x.length; i++) x[i] += mlp[i];                          // residual add
      if (capture) { cap.stream.push(Float32Array.from(x)); cap.attention.push(maps); cap.attnNorms.push(rowNorms(att, T)); cap.mlpNorms.push(rowNorms(mlp, T)); cap.streamNorms.push(rowNorms(x, T)); }
    }
    const hfin = layerNorm(x, T, w["ln_f.g"], w["ln_f.b"]);
    const logits = new Float64Array(V), o = (T - 1) * D;                          // only the LAST position is needed
    for (let v = 0; v < V; v++) { let s = 0; for (let i = 0; i < D; i++) s += hfin[o + i] * w["embed.tok"][v * D + i]; logits[v] = s; }
    return { logits, cap };
  }

  // ----------------------------------------------------------------- sampling
  function softmax(l) { const m = Math.max(...l), e = Array.from(l, v => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); }
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function sampleNext(logits, o, rand) {
    const raw = softmax(logits);
    if (o.temperature <= 0) { const k = raw.indexOf(Math.max(...raw)); return { id: k, raw, final: raw.map((_, i) => +(i === k)) }; }
    let sc = Array.from(logits, v => v / o.temperature);
    if (o.topK && o.topK < V) { const kth = sc.slice().sort((a, b) => b - a)[o.topK - 1]; sc = sc.map(v => (v >= kth ? v : -Infinity)); }
    if (o.topP && o.topP < 1) {
      const p = softmax(sc), order = p.map((_, i) => i).sort((a, b) => p[b] - p[a]); let c = 0, n = 0;
      while (n < order.length) { c += p[order[n++]]; if (c >= o.topP) break; }
      const keep = new Set(order.slice(0, n)); sc = sc.map((v, i) => (keep.has(i) ? v : -Infinity));
    }
    const fin = softmax(sc); let r = rand(), id = fin.length - 1;
    for (let i = 0; i < fin.length; i++) { r -= fin[i]; if (r <= 0) { id = i; break; } }
    return { id, raw, final: fin };
  }
  const top = (probs, fin, n) => probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]).slice(0, n)
    .map(i => ({ id: i, piece: piece(i), p_raw: probs[i], p_final: fin[i], p: probs[i] }));

  function generate(name, ids, o) {
    const w = weights(name), rand = rng(o.seed ?? 0), steps = []; ids = ids.slice();
    for (let n = 0; n < (o.maxNew ?? 48); n++) {
      const { logits } = forward(w, ids.slice(-CTX), false), s = sampleNext(logits, o, rand);
      steps.push({ id: s.id, piece: piece(s.id), candidates: top(s.final, s.final, 5) });
      if (s.id === special["<|end|>"]) break;
      ids.push(s.id);
    }
    return steps;
  }
  const textOf = steps => decode(steps.filter(s => s.id !== special["<|end|>"]).map(s => s.id)).trim();
  const formatPrompt = p => `<|user|>${p}<|assistant|>`;

  // ---------------------------------------------- interpretability helpers
  function unembedLast(w, stream, T) {      // "logit lens": pretend the stream is finished and read off a prediction
    const h = layerNorm(stream, T, w["ln_f.g"], w["ln_f.b"]), o = (T - 1) * D, l = new Float64Array(V);
    for (let v = 0; v < V; v++) { let s = 0; for (let i = 0; i < D; i++) s += h[o + i] * w["embed.tok"][v * D + i]; l[v] = s; }
    const p = softmax(l); return top(p, p, 5);
  }
  function neighbours(word, n = 8) {
    const w = weights("aligned"), ids = encode(word);
    if (ids.length !== 1) return { ids, pieces: ids.map(piece), list: [] };
    const E = w["embed.tok"], a = ids[0], na = Math.hypot(...E.subarray(a * D, a * D + D)), sims = [];
    for (let v = 0; v < V; v++) {
      if (v === a || piece(v).trim().length < 3) continue;
      let dot = 0, nb = 0; for (let i = 0; i < D; i++) { dot += E[a * D + i] * E[v * D + i]; nb += E[v * D + i] ** 2; }
      sims.push({ id: v, piece: piece(v), sim: dot / (na * Math.sqrt(nb)) });
    }
    return { ids, pieces: ids.map(piece), list: sims.sort((x, y) => y.sim - x.sim).slice(0, n) };
  }
  let pcaCache = null;
  function tokenMap() {                     // 2-D map of the whole vocabulary: top two principal components, by power iteration
    if (pcaCache) return pcaCache;
    const E = weights("aligned")["embed.tok"], mean = new Float64Array(D);
    for (let v = 0; v < V; v++) for (let i = 0; i < D; i++) mean[i] += E[v * D + i] / V;
    const X = new Float64Array(V * D);
    for (let v = 0; v < V; v++) for (let i = 0; i < D; i++) X[v * D + i] = E[v * D + i] - mean[i];
    const comps = [];
    for (let c = 0; c < 2; c++) {
      let u = new Float64Array(D).map((_, i) => Math.sin(i + c));
      for (let it = 0; it < 60; it++) {
        const y = new Float64Array(D);
        for (let v = 0; v < V; v++) { let s = 0; for (let i = 0; i < D; i++) s += X[v * D + i] * u[i]; for (let i = 0; i < D; i++) y[i] += s * X[v * D + i]; }
        for (const q of comps) { let d = 0; for (let i = 0; i < D; i++) d += y[i] * q[i]; for (let i = 0; i < D; i++) y[i] -= d * q[i]; }
        const n = Math.hypot(...y); u = y.map(z => z / n);
      }
      comps.push(u);
    }
    const pts = [];
    for (let v = 0; v < V; v++) { let x = 0, y = 0; for (let i = 0; i < D; i++) { x += X[v * D + i] * comps[0][i]; y += X[v * D + i] * comps[1][i]; } pts.push([x, y]); }
    return (pcaCache = pts);
  }
  function logitsFor(name, ids) { return forward(weights(name), ids.slice(-CTX), false).logits; }

  // ------------------------------------------ the whole journey, as one object
  function buildTrace(prompt, o, staticTrace) {
    const templated = formatPrompt(prompt), ids = encode(templated).slice(-CTX + 8), T = ids.length;
    const words = (prompt.match(WORD) || [" weather"]).slice().sort((a, b) => mergeSteps(b).length - mergeSteps(a).length);
    const { cap } = forward(weights("aligned"), ids, true);
    const grid = (f) => Array.from({ length: T }, (_, t) => Array.from({ length: 16 }, (_, i) => +f(t, i).toFixed(3)));
    const w = weights("aligned"), greedy = { temperature: 0 };
    const first = name => { const s = sampleNext(forward(weights(name), ids, false).logits, { temperature: 1 }, rng(0)); return top(s.raw, s.raw, 5); };
    const gen = generate("aligned", ids, o);
    const lens = cap.stream.map(st => unembedLast(w, st, T));
    const lastVec = cap.stream.map(st => Array.from(st.subarray((T - 1) * D, T * D)).map(v => +v.toFixed(3)));
    return Object.assign({}, staticTrace, {
      prompt, live: true, logit_lens: lens, last_vectors: lastVec, n_merges: M.merges.length, merge_counts: M.merge_counts || null,
      input: { text: prompt, n_chars: [...prompt].length, bytes: [...prompt].map(c => new TextEncoder().encode(c).join("+")) },
      tokens: { templated, ids, pieces: ids.map(piece), vocab_size: V, merge_demo: { word: words[0], steps: mergeSteps(words[0]).map(s => s.map(piece)) } },
      embedding: { shape: [T, D], token: grid((t, i) => w["embed.tok"][ids[t] * D + i]), position: grid((t, i) => w["embed.pos"][t * D + i]), sum: grid((t, i) => cap.embed[t * D + i]) },
      transformer: { stream_norms: cap.streamNorms, attn_norms: cap.attnNorms, mlp_norms: cap.mlpNorms },
      attention: cap.attention, generation: gen, output: textOf(gen),
      sampling_demo: { sft: first("sft"), aligned: first("aligned") },
      answers: { base: textOf(generate("base", ids, Object.assign({ maxNew: 40 }, greedy))), sft: textOf(generate("sft", ids, greedy)), aligned: textOf(generate("aligned", ids, greedy)) },
    });
  }

  return { encode, encodeUpTo, decode, piece, mergeSteps, forward, weights, generate, textOf, formatPrompt, buildTrace, sampleNext,
           neighbours, tokenMap, logitsFor, softmax, config: cfg, nMerges: M.merges.length, mergeRule: k => [piece(M.merges[k][0]), piece(M.merges[k][1]), piece(256 + k), (M.merge_counts || [])[k]] };
  }
  root.createEngine = createEngine;
  if (root.MODEL) root.Engine = createEngine(root.MODEL);
})(typeof window !== "undefined" ? window : globalThis);
