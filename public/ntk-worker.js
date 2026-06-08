/**
 * Phase 40 next-4-b session 3 — NTK-Mirror browser worker.
 *
 * Federated SPSA worker for the gate-controller protocol (`/api/ntk/*`)
 * that actually runs Qwen-0.5B-Instruct forward IN THE BROWSER TAB via
 * onnxruntime-web, with our NTK-Mirror gates injected as a forward input
 * (`gate_mults: [24, 896] float32`).
 *
 * Architecture mirror of `scripts/ntk-verifier.py` line-by-line:
 *
 *   bootstrap → fetch the gate-selection artifact + snapshot raw[K=5000]
 *   load the int8 Qwen+gates ONNX via OPFS (cached across reloads)
 *   per round:
 *     poll  /api/ntk/tick (handles other workers' applied flips)
 *     reconcile applied_since (replay SPSA updates on local raw[K])
 *     compute loss_before via forward(token_ids, attn_mask, gate_mults(raw))
 *     for T=4 SPSA trials:
 *       seed_t  = u32 random
 *       u_t     = mulberry32+Box-Muller(seed_t)
 *       g_t     = (forward(θ+εu) − forward(θ−εu)) / 2ε
 *       loss_at = forward(θ − η·g·u)
 *       claimed_Δ_t = loss_at − loss_before
 *     submit best (seed, scalar_g, claimed_Δ) WITH audit_loss_before
 *     server runs Phase 39 sym-AIMD η + byzantine real_Δ check on the
 *     PRIOR round's winner (one-round lag).
 *
 * Tokenizer: not needed in browser. The math-corpus tokens (Qwen-0.5B's
 * tokenizer, 4 examples × 32 tokens) are pre-computed and baked in at
 * the top of this file — same training corpus the Python verifier uses.
 *
 * Hosting:
 *   local dev: wrangler dev serves /data/qwen05b-with-gates-int8.onnx
 *              at any size (file is on disk, not bundled).
 *   production: 866 MB > CF assets limit. Upload to HuggingFace Hub
 *               (free public CDN) and point ONNX_URL at the HF URL.
 *               See docs/PHASE_40_NEXT4B_QWEN_ONNX.md.
 */

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs";

// Tell ORT-web where to find its wasm artifacts (matched to the import URL).
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.numThreads = 1;  // we're not cross-origin-isolated in CF assets

// ─── constants matching src/tournament-ntk.ts and src/ntk-gate.ts ──

const ARTIFACT_URL = "/data/qwen05b-math-gates-k5000.bin";

// The 866 MB int8 ONNX can't live in public/ — wrangler caps assets at
// 25 MiB. For LOCAL dev: serve it from a sibling directory via a CORS-
// friendly static server (one-liner below). For PROD: upload to HF Hub
// or R2 and point this URL there.
//
//   # in a separate terminal:
//   cd ~/postnet-cf-onnx
//   npx http-server -p 8788 --cors -c-1 .
//   # then the worker on http://localhost:8787 can fetch from :8788.
//
// Override via URL query, e.g. ?onnx=https://huggingface.co/.../model.onnx
const _qsOnnx = (() => {
  if (typeof location === "undefined") return null;
  return new URLSearchParams(location.search).get("onnx");
})();
// Phase 40 next-4-b session 5: optimum-cli + scripts/inject-gates-onnx.py
// produced a 2.4 GB fp32 ONNX that ORT-web aborted on (load step), and
// Chrome's ArrayBuffer max ~2 GB prevented loading the sidecar anyway.
// We int8-quantize-dynamic the gated ONNX → 994 MB single file (no sidecar),
// fits in one ArrayBuffer, and ORT-web 1.22 loads it.
const ONNX_URL = _qsOnnx || "http://localhost:8788/qwen05b-with-gates-optimum-int8.onnx";
const ONNX_DATA_URL = null;       // single-file int8 — no sidecar needed
const ONNX_OPFS_NAME = "qwen05b-with-gates-optimum-int8.onnx";

const K = 5000;
const N_LAYERS = 24;
const HIDDEN_SIZE = 896;
const MAX_LOG_GATE = 0.05;
const TRIALS_PER_REPORT = 2;        // 2 for browser to keep per-round cost manageable
const POLL_DELAY_MS = 200;

// Hyperparams pulled from server on every /tick.
let EPSILON = 0.005;
let ETA = 0.001;

// ─── baked tokenized inputs (Qwen2.5-0.5B tokenizer; matches the math
//     corpus the Python verifier uses) ────────────────────────────────

const TOKEN_IDS = [
  [31198, 25, 220, 16, 19, 488, 220, 17, 22, 284, 17607, 36842, 25, 2691, 6174, 25, 220, 19, 10, 22, 28, 16, 16, 11, 3270, 220, 16, 6777, 220, 16, 13, 350],
  [31198, 25, 220, 18, 21, 488, 220, 16, 23, 284, 17607, 36842, 25, 2691, 6174, 25, 220, 21, 10, 23, 28, 16, 19, 11, 3270, 220, 19, 6777, 220, 16, 13, 350],
  [31198, 25, 220, 19, 20, 488, 220, 17, 24, 284, 17607, 36842, 25, 2691, 6174, 25, 220, 20, 10, 24, 28, 16, 19, 11, 3270, 220, 19, 6777, 220, 16, 13, 350],
  [31198, 25, 220, 21, 18, 488, 220, 17, 23, 284, 17607, 36842, 25, 2691, 6174, 25, 220, 18, 10, 23, 28, 16, 16, 11, 3270, 220, 16, 6777, 220, 16, 13, 350],
];
const BATCH = TOKEN_IDS.length;
const SEQ_LEN = TOKEN_IDS[0].length;
// Flatten to row-major Int64 typed-arrays (ORT expects BigInt64 for int64).
const inputIdsFlat = new BigInt64Array(BATCH * SEQ_LEN);
const attnMaskFlat = new BigInt64Array(BATCH * SEQ_LEN);
// optimum-cli's text-generation export expects position_ids too.
const positionIdsFlat = new BigInt64Array(BATCH * SEQ_LEN);
for (let b = 0; b < BATCH; b++) {
  for (let t = 0; t < SEQ_LEN; t++) {
    inputIdsFlat[b * SEQ_LEN + t] = BigInt(TOKEN_IDS[b][t]);
    attnMaskFlat[b * SEQ_LEN + t] = 1n;
    positionIdsFlat[b * SEQ_LEN + t] = BigInt(t);
  }
}
// Labels for causal-LM loss: predict token t+1 given tokens [0..t].
// HF convention: labels = input_ids, then shift inside the loss.
// We'll compute the shift in JS; -100 marks "ignore" positions.
const labelsFlat = new Int32Array(BATCH * SEQ_LEN);
for (let i = 0; i < BATCH * SEQ_LEN; i++) labelsFlat[i] = Number(inputIdsFlat[i]);

// ─── mulberry32 + Box-Muller (bit-identical to src/ntk-gate.ts) ──

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function reconstructPerturbation(seed, out) {
  const rng = mulberry32(seed);
  for (let i = 0; i < K; i += 2) {
    let u1 = 0, u2 = 0;
    while (u1 === 0) u1 = rng();
    while (u2 === 0) u2 = rng();
    const mag = Math.sqrt(-2 * Math.log(u1));
    const ang = 2 * Math.PI * u2;
    out[i] = mag * Math.cos(ang);
    if (i + 1 < K) out[i + 1] = mag * Math.sin(ang);
  }
}

// ─── parse the gate-selection artifact (matches src/ntk-gate.ts) ──

async function loadGateArtifact() {
  const r = await fetch(ARTIFACT_URL);
  if (!r.ok) throw new Error(`gate artifact ${r.status}`);
  const buf = await r.arrayBuffer();
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "NTKG") throw new Error(`bad gate-artifact magic ${magic}`);
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`bad gate-artifact version ${version}`);
  const Kart = view.getUint32(8, true);
  if (Kart !== K) throw new Error(`K mismatch: artifact=${Kart} expected=${K}`);
  let off = 32;
  const layerIdx = new Uint16Array(buf, off, K).slice();
  off += K * 2;
  const channelIdx = new Uint16Array(buf, off, K).slice();
  return { layerIdx, channelIdx, bytes: buf.byteLength };
}

// gate_mults: [N_LAYERS, HIDDEN_SIZE] float32, initialized to 1.0
// then for each gate j: mults[layer_idx[j], channel_idx[j]] = exp(MAX_LOG_GATE * tanh(raw[j])).
const gateMultsBuf = new Float32Array(N_LAYERS * HIDDEN_SIZE);
function fillGateMults(raw, artifact) {
  gateMultsBuf.fill(1.0);
  for (let j = 0; j < K; j++) {
    const l = artifact.layerIdx[j];
    const c = artifact.channelIdx[j];
    gateMultsBuf[l * HIDDEN_SIZE + c] = Math.exp(MAX_LOG_GATE * Math.tanh(raw[j]));
  }
}

// ─── OPFS-cached ONNX loader (so the 866 MB download happens once) ──

// Chrome's OPFS createWritable.write() rejects buffers above ~2 GB
// (FileSystemSyncAccessHandle error). For files that big, we skip the
// cache and just keep them in memory — slower across reloads, but the
// alternative is a partial-write that fails on read.
const OPFS_MAX_BYTES = 1.5 * 1024 * 1024 * 1024;   // 1.5 GB safety margin

async function _loadOneToOpfs(url, opfsName, progressFn, label) {
  const root = await navigator.storage.getDirectory();
  try {
    const fh = await root.getFileHandle(opfsName, { create: false });
    const file = await fh.getFile();
    if (file.size > 1024) {
      progressFn(`OPFS hit ${label}: ${(file.size / 1024 / 1024).toFixed(0)} MB`);
      return await file.arrayBuffer();
    }
  } catch {
    // fall through to download
  }
  progressFn(`downloading ${label}...`);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${label} fetch ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || 0;
  let buf;
  // Prefer Response.bytes() (Chrome 116+, Firefox 130+) — returns a single
  // Uint8Array directly without going through chunks→Blob→arrayBuffer,
  // which halves peak memory during the download of a 1 GB+ ONNX.
  if (typeof resp.bytes === "function") {
    const bytes = await resp.bytes();
    buf = bytes.buffer;
    progressFn(`${label}: ${(bytes.length / 1024 / 1024).toFixed(0)} MB (single-allocation)`);
  } else {
    // Fallback: streamed chunks → Blob → ArrayBuffer (older browsers; uses
    // ~2× peak memory during the final blob.arrayBuffer() call).
    const chunks = [];
    const reader = resp.body.getReader();
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) progressFn(`${label}: ${(received / 1024 / 1024).toFixed(0)} / ${(total / 1024 / 1024).toFixed(0)} MB (${(100 * received / total).toFixed(0)}%)`);
      else       progressFn(`${label}: ${(received / 1024 / 1024).toFixed(0)} MB`);
    }
    const blob = new Blob(chunks);
    buf = await blob.arrayBuffer();
  }
  if (buf.byteLength > OPFS_MAX_BYTES) {
    progressFn(`${label} too large for OPFS (${(buf.byteLength / 1024 / 1024).toFixed(0)} MB > ${(OPFS_MAX_BYTES / 1024 / 1024).toFixed(0)} MB); keeping in memory only`);
    return buf;
  }
  try {
    const fh = await root.getFileHandle(opfsName, { create: true });
    const w = await fh.createWritable();
    await w.write(buf);
    await w.close();
    progressFn(`cached ${label} to OPFS (${(buf.byteLength / 1024 / 1024).toFixed(0)} MB)`);
  } catch (e) {
    progressFn(`OPFS write of ${label} failed (${e.message}); in memory only`);
  }
  return buf;
}

// Single-file int8 ONNX after our int8 quantization pass.
async function loadOnnxToOpfs(progressFn) {
  const graph = await _loadOneToOpfs(ONNX_URL, ONNX_OPFS_NAME, progressFn, "model");
  return { graph, data: null };
}

let ortSession = null;

// Backend selection. WebGPU is 3-5× faster on supported hardware; WASM
// is the universal fallback. Override with ?backend=wasm or ?backend=webgpu.
function pickBackends() {
  const qs = (typeof location !== "undefined")
    ? new URLSearchParams(location.search).get("backend") : null;
  if (qs === "wasm") return [["wasm", "WASM (forced)"]];
  if (qs === "webgpu") return [["webgpu", "WebGPU (forced)"]];
  // Default order: try WebGPU, fall back to WASM.
  if (typeof navigator !== "undefined" && navigator.gpu) {
    return [["webgpu", "WebGPU"], ["wasm", "WASM (fallback)"]];
  }
  return [["wasm", "WASM"]];
}

async function initOrtSession(onnxBufs, progressFn) {
  const candidates = pickBackends();
  let lastErr = null;
  for (const [ep, label] of candidates) {
    progressFn(`initializing ORT session (${label})...`);
    const opts = {
      executionProviders: [ep],
      graphOptimizationLevel: "all",
    };
    if (onnxBufs.data) {
      opts.externalData = [{ data: new Uint8Array(onnxBufs.data), path: "weights.bin" }];
    }
    try {
      ortSession = await ort.InferenceSession.create(new Uint8Array(onnxBufs.graph), opts);
      progressFn(`ORT ready (${label}). inputs=${ortSession.inputNames.join(", ")}, outputs=${ortSession.outputNames.join(", ")}`);
      return;
    } catch (e) {
      lastErr = e;
      progressFn(`${label} failed: ${(e?.message || e).toString().slice(0, 100)}`);
    }
  }
  throw lastErr || new Error("no ORT backend available");
}

// ─── forward + cross-entropy loss given current raw[K] ──

async function forwardLoss(raw, artifact) {
  fillGateMults(raw, artifact);
  const feeds = {
    input_ids:      new ort.Tensor("int64",   inputIdsFlat,    [BATCH, SEQ_LEN]),
    attention_mask: new ort.Tensor("int64",   attnMaskFlat,    [BATCH, SEQ_LEN]),
    position_ids:   new ort.Tensor("int64",   positionIdsFlat, [BATCH, SEQ_LEN]),
    gate_mults:     new ort.Tensor("float32", gateMultsBuf,    [N_LAYERS, HIDDEN_SIZE]),
  };
  const out = await ortSession.run(feeds);
  const logits = out.logits.data;                // Float32Array, [BATCH, SEQ_LEN, V]
  const V = out.logits.dims[2];

  // causal-LM loss: for each position t, predict label[t+1] from logits[t].
  // mean over non-masked positions.
  let totalLoss = 0;
  let count = 0;
  for (let b = 0; b < BATCH; b++) {
    for (let t = 0; t < SEQ_LEN - 1; t++) {
      const tgt = labelsFlat[b * SEQ_LEN + t + 1];
      if (tgt < 0) continue;
      const base = (b * SEQ_LEN + t) * V;
      // softmax + nll: max-subtract for stability
      let mx = -Infinity;
      for (let v = 0; v < V; v++) if (logits[base + v] > mx) mx = logits[base + v];
      let sum = 0;
      for (let v = 0; v < V; v++) sum += Math.exp(logits[base + v] - mx);
      const loss = -(logits[base + tgt] - mx - Math.log(sum + 1e-7));
      totalLoss += loss;
      count++;
    }
  }
  return totalLoss / Math.max(count, 1);
}

// ─── UI bindings ──

const workerId = `ntk-browser-${Math.random().toString(36).slice(2, 8)}`;
const $ = (id) => document.getElementById(id);
const widEl = $("wid"), roundEl = $("round"), poolEl = $("pool");
const peersEl = $("peers"), lossEl = $("loss"), etaEl = $("eta");
const arEl = $("ar"), bwEl = $("bw"), thetaNormEl = $("theta-norm");
const logEl = $("log"), progEl = $("progress");
const joinBtn = $("join"), resetBtn = $("reset");
const chartCanvas = $("chart");
const ctx = chartCanvas.getContext("2d");
widEl.textContent = workerId;

let running = false;
let history = [];
let localTheta = null;        // raw[K]
let localRound = -1;
let bytesUp = 0, bytesDown = 0;
let ws = null;
let wsConnected = false;
let artifact = null;
const uScratch = new Float32Array(K);
const thetaPlus = new Float32Array(K);
const thetaMinus = new Float32Array(K);
const thetaStep = new Float32Array(K);
let currentEta = ETA;

function log(s) {
  const stamp = new Date().toLocaleTimeString();
  logEl.textContent = `[${stamp}] ${s}\n` + logEl.textContent.slice(0, 4000);
}
function setProgress(s) {
  if (progEl) progEl.textContent = s;
  log(s);
}
function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(2) + " MB";
}
function updateBwStat() {
  if (bwEl) bwEl.textContent = `↑${fmtBytes(bytesUp)} ↓${fmtBytes(bytesDown)}`;
}
async function trackedFetch(url, init) {
  const body = init && init.body ? init.body : "";
  bytesUp += new TextEncoder().encode(body).length;
  const r = await fetch(url, init);
  const text = await r.text();
  bytesDown += new TextEncoder().encode(text).length;
  updateBwStat();
  if (!r.ok) throw new Error(`coord ${r.status}: ${text.slice(0, 100)}`);
  return JSON.parse(text);
}

// ─── network glue ──

async function bootstrapSnapshot() {
  const meta = await trackedFetch("/api/ntk/snapshot");
  if (typeof meta.epsilon === "number") EPSILON = meta.epsilon;
  if (typeof meta.eta === "number") { ETA = meta.eta; currentEta = meta.eta; }
  const buffers = await Promise.all(meta.shards.map(async s => {
    const r = await fetch(s.url);
    const buf = await r.arrayBuffer();
    bytesDown += buf.byteLength;
    return { shard: s.shard, headerSize: s.shard === 0 ? 8 : 0, buf };
  }));
  updateBwStat();
  buffers.sort((a, b) => a.shard - b.shard);
  const headView = new DataView(buffers[0].buf);
  localRound = headView.getUint32(0, true);
  const p = headView.getUint32(4, true);
  if (p !== K) throw new Error(`K mismatch: server=${p} client=${K}`);
  localTheta = new Float32Array(K);
  for (const b of buffers) {
    const start = b.shard * (meta.shard_size_floats || (b.buf.byteLength - b.headerSize) / 4);
    localTheta.set(new Float32Array(b.buf, b.headerSize), start);
  }
  log(`snapshot @ R${localRound} · ε=${EPSILON} η=${currentEta.toExponential(2)}`);
}

async function tickPoll() {
  return await trackedFetch("/api/ntk/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ worker_id: workerId, since_round: localRound }),
  });
}
async function submitBest(round, seed, scalar_g, delta, audit_loss_before) {
  return await trackedFetch("/api/ntk/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      worker_id: workerId,
      round, seed, scalar_g, delta,
      since_round: localRound,
      audit_loss_before,
    }),
  });
}
function applyDelta(applied) {
  reconstructPerturbation(applied.seed, uScratch);
  // Phase 40 next-7: replay with the flip's server-stamped η, not currentEta,
  // so this replica stays bit-identical to the server. Replaying with a stale
  // currentEta is what drifted the 2-worker replicas apart (‖θ‖ 0.597 vs 0.408)
  // and false-quarantined both honest workers.
  const flipEta = typeof applied.eta === "number" ? applied.eta : currentEta;
  const k = flipEta * applied.scalar_g;
  for (let i = 0; i < K; i++) localTheta[i] -= k * uScratch[i];
}
async function reconcile(resp) {
  if (typeof resp.epsilon === "number") EPSILON = resp.epsilon;
  if (typeof resp.eta === "number") { ETA = resp.eta; currentEta = resp.eta; }
  if (resp.round === localRound) return;
  if (typeof resp.oldest_applied_round === "number"
      && resp.oldest_applied_round > localRound + 1) {
    log("history truncated — re-bootstrap");
    await bootstrapSnapshot();
    return;
  }
  if (Array.isArray(resp.applied_since)) {
    for (const flip of resp.applied_since) {
      if (flip.round >= localRound) applyDelta(flip);
    }
  }
  localRound = resp.round;
}
function updateStats(s) {
  roundEl.textContent = s.round;
  poolEl.textContent = `${s.proposals}/${s.target}`;
  peersEl.textContent = s.joined;
  if (typeof s.last_loss === "number") lossEl.textContent = s.last_loss.toFixed(4);
  if (typeof s.eta === "number") etaEl.textContent = s.eta.toExponential(2);
  if (typeof s.accept_rate === "number") arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
}

function openWebSocket() {
  try {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/api/ntk/ws`);
    ws.addEventListener("open", () => { wsConnected = true; log("ws connected"); });
    ws.addEventListener("message", (e) => {
      bytesDown += e.data.length;
      updateBwStat();
      try {
        const m = JSON.parse(e.data);
        if (m.type === "advance" && localTheta) {
          if (typeof m.eta === "number") { ETA = m.eta; currentEta = m.eta; }
          if (m.applied && m.applied.round === localRound) applyDelta(m.applied);
          localRound = m.round;
        }
      } catch {}
    });
    ws.addEventListener("close", () => { wsConnected = false; log("ws closed — back to polling"); });
    ws.addEventListener("error", () => { wsConnected = false; });
  } catch (e) {
    log(`ws failed: ${e.message}`);
  }
}

function drawChart() {
  const w = chartCanvas.width, h = chartCanvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;
  const losses = history.filter(p => p.loss > 0).map(p => p.loss);
  if (losses.length < 2) return;
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const range = Math.max(hi - lo, 0.0001);
  const padX = 40, padY = 20;
  ctx.strokeStyle = "#d4d2cc"; ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, w - padX - 14, h - padY * 2);
  ctx.strokeStyle = "#d6502c"; ctx.lineWidth = 2;
  ctx.beginPath();
  let drawn = false;
  for (let i = 0; i < history.length; i++) {
    if (!(history[i].loss > 0)) continue;
    const x = padX + (i / Math.max(history.length - 1, 1)) * (w - padX - 14);
    const y = (h - padY) - ((history[i].loss - lo) / range) * (h - padY * 2);
    if (!drawn) { ctx.moveTo(x, y); drawn = true; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#666"; ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(hi.toFixed(4), 4, padY + 8);
  ctx.fillText(lo.toFixed(4), 4, h - padY);
}

async function pullState() {
  try {
    const r = await fetch("/api/ntk/state");
    const s = await r.json();
    history = (s.history || []).filter(h => h.round >= 0);
    drawChart();
    if (!running) {
      roundEl.textContent = s.round;
      poolEl.textContent = `${s.proposals}/${s.target}`;
      peersEl.textContent = (s.joined || []).length;
      if (typeof s.last_loss === "number") lossEl.textContent = s.last_loss.toFixed(4);
      if (typeof s.eta === "number") etaEl.textContent = s.eta.toExponential(2);
      arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
      if (s.theta_stats) thetaNormEl.textContent = s.theta_stats.l2_norm.toFixed(4);
    }
  } catch {}
}

// ─── main loop ──

async function spsaTrial(rng, lossBefore) {
  const seed = (rng() * 0xFFFFFFFF) >>> 0;
  reconstructPerturbation(seed, uScratch);
  for (let i = 0; i < K; i++) {
    thetaPlus[i] = localTheta[i] + EPSILON * uScratch[i];
    thetaMinus[i] = localTheta[i] - EPSILON * uScratch[i];
  }
  const lossPlus = await forwardLoss(thetaPlus, artifact);
  const lossMinus = await forwardLoss(thetaMinus, artifact);
  const scalar_g = (lossPlus - lossMinus) / (2 * EPSILON);
  const kStep = currentEta * scalar_g;
  for (let i = 0; i < K; i++) thetaStep[i] = localTheta[i] - kStep * uScratch[i];
  const lossAt = await forwardLoss(thetaStep, artifact);
  return { seed, scalar_g, delta: lossAt - lossBefore };
}

// Attack mode: enabled via ?attack=1 URL param OR the #attack checkbox.
// When on, this worker fabricates random seeds with claimed_Δ = -10 and
// skips the entire ONNX/ORT path (saves ~2 GB RAM per attacker tab).
// Server's Phase 39 byzantine check catches the pattern within ~10 wins.
function readAttackFlag() {
  if (typeof location !== "undefined") {
    if (new URLSearchParams(location.search).get("attack") === "1") return true;
  }
  const el = typeof document !== "undefined" ? document.getElementById("attack") : null;
  return !!(el && el.checked);
}

async function runForever() {
  if (readAttackFlag()) {
    // Attack mode short-circuit: no ONNX load, no ORT, no gate artifact,
    // no snapshot. Saves ~2 GB of RAM per attacker tab so the honest
    // workers can breathe. Attacker just needs to know the current
    // round number to submit valid proposals.
    log("attack mode: skipping ONNX + artifact load (~2 GB saved)");
    setProgress("attack mode (no model load)");
    const pulled = await tickPoll();
    if (typeof pulled.round === "number") localRound = pulled.round;
    if (typeof pulled.eta === "number") { ETA = pulled.eta; currentEta = pulled.eta; }
    openWebSocket();
    while (running) {
      try {
        const seedBase = ((localRound + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
        const rng = mulberry32(seedBase);
        const fakeSeed = (rng() * 0xFFFFFFFF) >>> 0;
        // Phase 40 next-6: attacker MUST NOT send audit_loss_before — running
        // no forward passes means no loss to report. Previously sent 0,
        // which corrupted the server's lastLoss baseline AND let one attacker
        // cross-close pending audits for OTHER attackers (Sybil amplifier).
        // Server-side guard rejects audit_loss_before <= 0; this matches.
        const reported = await submitBest(localRound, fakeSeed, 1.0, -10, undefined);
        if (typeof reported.eta === "number") { ETA = reported.eta; currentEta = reported.eta; }
        if (reported.quarantined) {
          log(`⚠ QUARANTINED — server detected fraud (this was the point)`);
        } else if (reported.advanced) {
          log(`[attacker] R${localRound} → R${reported.round}`);
        }
        if (typeof reported.round === "number") localRound = reported.round;
      } catch (e) {
        log(`[attacker] err: ${e.message}`);
      }
      await new Promise(r => setTimeout(r, POLL_DELAY_MS));
    }
    return;
  }

  // Heavy init (honest workers only): artifact, ONNX, ORT session, snapshot.
  artifact = await loadGateArtifact();
  log(`gate artifact loaded (${(artifact.bytes / 1024).toFixed(1)} KB)`);
  const onnxBufs = await loadOnnxToOpfs(setProgress);
  await initOrtSession(onnxBufs, setProgress);
  await bootstrapSnapshot();
  openWebSocket();

  // First forward to warm + report baseline.
  const baseLoss = await forwardLoss(localTheta, artifact);
  log(`baseline loss @ θ=current: ${baseLoss.toFixed(4)}`);
  if (thetaNormEl) {
    let s = 0; for (let i = 0; i < K; i++) s += localTheta[i] * localTheta[i];
    thetaNormEl.textContent = Math.sqrt(s).toFixed(4);
  }

  while (running) {
    try {
      if (!wsConnected) {
        const pulled = await tickPoll();
        updateStats(pulled);
        await reconcile(pulled);
      }
      const seedBase = ((localRound + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
      const rng = mulberry32(seedBase);
      let lossBefore;
      let best;
      if (readAttackFlag()) {
        // Byzantine: skip the forward, fabricate (seed, scalar_g, delta).
        // claimed_Δ = -10 is way below the no-op threshold, so server's
        // tournament will pick this. After apply, real_Δ ≈ 0 (random
        // perturbation) so the per-round fraud bit fires; ~10 wins to
        // quarantine.
        const fakeSeed = (rng() * 0xFFFFFFFF) >>> 0;
        lossBefore = 0;                    // not used by server's audit math
        best = { seed: fakeSeed, scalar_g: 1.0, delta: -10 };
      } else {
        lossBefore = await forwardLoss(localTheta, artifact);
        best = null;
        for (let t = 0; t < TRIALS_PER_REPORT; t++) {
          const trial = await spsaTrial(rng, lossBefore);
          if (!best || trial.delta < best.delta) best = trial;
        }
      }
      const reported = await submitBest(localRound, best.seed, best.scalar_g, best.delta, lossBefore);
      updateStats(reported);
      await reconcile(reported);
      if (reported.quarantined) {
        log(`⚠ QUARANTINED · proposals dropped — defense detected fraud`);
      } else if (reported.advanced) {
        log(`R${localRound - 1} → R${localRound} · Δ ${best.delta.toFixed(4)} · η ${currentEta.toExponential(2)}`);
        if (thetaNormEl) {
          let s = 0; for (let i = 0; i < K; i++) s += localTheta[i] * localTheta[i];
          thetaNormEl.textContent = Math.sqrt(s).toFixed(4);
        }
      } else if (reported.rejected) {
        log(`stale @ R${localRound}`);
      }
    } catch (e) {
      log(`error: ${e.message}`);
    }
    await new Promise(r => setTimeout(r, POLL_DELAY_MS));
  }
}

joinBtn.addEventListener("click", () => {
  if (running) return;
  running = true;
  joinBtn.disabled = true;
  joinBtn.textContent = "Joined";
  log(`joining as ${workerId}`);
  pullState();
  runForever().catch(e => {
    running = false;
    joinBtn.disabled = false;
    joinBtn.textContent = "Join NTK swarm";
    log(`fatal: ${e.message || e}`);
  });
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset NTK coordinator? All progress lost.")) return;
  await fetch("/api/ntk/reset", { method: "POST" });
  localTheta = null;
  localRound = -1;
  history = [];
  bytesUp = 0; bytesDown = 0;
  updateBwStat();
  drawChart();
  log("reset");
});

pullState();
setInterval(pullState, 5000);
