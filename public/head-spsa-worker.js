/**
 * Phase 40 next-4-a — browser worker for the head-classifier (Phase 38/39).
 *
 * Completes the demo arc: Phase 38 shipped the federated DO, Phase 39 added
 * adaptive η, but workers were always Node-side via verifier scripts. This
 * is the browser-tab worker — open the page, click Join, federate-train an
 * AG News head classifier with anyone else who has the URL open.
 *
 * Model: 2-layer MLP head over precomputed MiniLM-L6-v2 features.
 *   D = 384 (MiniLM dim), H = 128 (hidden), K = 4 (classes)
 *   P = D*H + H + H*K + K = 49 796 trainable scalars.
 *   Features pre-baked in /data/agnews-mini.bin (Phase 38 artifact, 154 KB).
 *
 * Protocol: SPSA-tournament with Phase 39's symmetric-AIMD adaptive η.
 * Server is src/tournament-head-spsa-adaptive.ts. Per-proposal wire
 * format = 20 bytes (seed + scalar_g + claimed_Δ); η rides every /tick
 * response so the browser stays synced with the server's current step size.
 *
 * Why this is *not* the NTK Qwen browser worker (Phase 40 next-4-b):
 *   - The head-classifier is tiny (P=50K) and runs in pure JS in ms.
 *   - A Qwen-0.5B browser worker needs ONNX-graph mod or neuropulse WGSL
 *     to inject NTK-Mirror gates inside a 500M-param forward — a real
 *     multi-session engineering effort. Deferred.
 */

const D = 384, H = 128, K = 4;
const P_FC1 = D * H, P_B1 = H, P_FC2 = H * K, P_B2 = K;
const P = P_FC1 + P_B1 + P_FC2 + P_B2;        // 49 796
const FC1_OFF = 0;
const B1_OFF = FC1_OFF + P_FC1;
const FC2_OFF = B1_OFF + P_B1;
const B2_OFF = FC2_OFF + P_FC2;

const TRAIN_N = 75;
const NUM_SHARDS = 3;
const TRIALS_PER_REPORT = 4;
const POLL_DELAY_MS = 80;
const CLASS_NAMES = ["World", "Sports", "Business", "Sci-Tech"];

// Hyperparams. η updates from server on every /tick.
let EPSILON = 0.005;
let ETA = 0.001;

// ─── deterministic perturbation (matches src/tournament-head-spsa-adaptive.ts) ─

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
  for (let i = 0; i < P; i += 2) {
    let u1 = 0, u2 = 0;
    while (u1 === 0) u1 = rng();
    while (u2 === 0) u2 = rng();
    const mag = Math.sqrt(-2 * Math.log(u1));
    const ang = 2 * Math.PI * u2;
    out[i] = mag * Math.cos(ang);
    if (i + 1 < P) out[i + 1] = mag * Math.sin(ang);
  }
}

// ─── head-classifier math (matches src/head-model.ts) ─

function forward(theta, x, logits) {
  const h = new Float32Array(H);
  for (let j = 0; j < H; j++) h[j] = theta[B1_OFF + j];
  for (let i = 0; i < D; i++) {
    const xi = x[i]; if (xi === 0) continue;
    const row = FC1_OFF + i * H;
    for (let j = 0; j < H; j++) h[j] += xi * theta[row + j];
  }
  for (let j = 0; j < H; j++) if (h[j] < 0) h[j] = 0;
  for (let k = 0; k < K; k++) logits[k] = theta[B2_OFF + k];
  for (let j = 0; j < H; j++) {
    const hj = h[j]; if (hj === 0) continue;
    const row = FC2_OFF + j * K;
    for (let k = 0; k < K; k++) logits[k] += hj * theta[row + k];
  }
}
function classLoss(theta, features, labels, start, end) {
  const logits = new Float32Array(K);
  const x = new Float32Array(D);
  let loss = 0;
  for (let n = start; n < end; n++) {
    for (let i = 0; i < D; i++) x[i] = features[n * D + i];
    forward(theta, x, logits);
    let mx = logits[0];
    for (let k = 1; k < K; k++) if (logits[k] > mx) mx = logits[k];
    let sum = 0;
    for (let k = 0; k < K; k++) sum += Math.exp(logits[k] - mx);
    const tgt = labels[n];
    loss += -(logits[tgt] - mx - Math.log(sum + 1e-7));
  }
  return loss / Math.max(end - start, 1);
}

// ─── shard the train set deterministically by worker_id ─

function shardForWorker(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  const s = h % NUM_SHARDS;
  const span = Math.floor(TRAIN_N / NUM_SHARDS);
  const start = s * span;
  const end = s === NUM_SHARDS - 1 ? TRAIN_N : (s + 1) * span;
  return { start, end, idx: s };
}

// ─── feature dataset (/data/agnews-mini.bin format from extract-agnews-features.mjs) ─

async function loadDataset() {
  const r = await fetch("/data/agnews-mini.bin");
  if (!r.ok) throw new Error(`agnews-mini.bin ${r.status}`);
  const buf = await r.arrayBuffer();
  const view = new DataView(buf);
  const N = view.getUint32(0, true);
  const headerBytes = 16;
  return {
    N,
    features: new Float32Array(buf, headerBytes, N * D).slice(),
    labels:   new Uint8Array (buf, headerBytes + N * D * 4, N).slice(),
    bytes: buf.byteLength,
  };
}

// ─── attack mode (mirrors lm.html convention) ─

function readAttackFlag() {
  if (typeof location !== "undefined") {
    if (new URLSearchParams(location.search).get("attack") === "1") return true;
  }
  const el = typeof document !== "undefined" ? document.getElementById("attack") : null;
  return !!(el && el.checked);
}

// ─── UI bindings ─

const workerId = `head-${Math.random().toString(36).slice(2, 8)}`;
const $ = (id) => document.getElementById(id);
const widEl = $("wid"), shardEl = $("shard"), roundEl = $("round"), poolEl = $("pool");
const peersEl = $("peers"), lossEl = $("loss"), accEl = $("acc"), etaEl = $("eta");
const arEl = $("ar"), bwEl = $("bw"), logEl = $("log");
const chartCanvas = $("chart");
const joinBtn = $("join"), resetBtn = $("reset");
const ctx = chartCanvas.getContext("2d");
const myShard = shardForWorker(workerId);
widEl.textContent = workerId;
shardEl.textContent = `shard ${myShard.idx} (examples ${myShard.start}–${myShard.end - 1})`;

let running = false;
let history = [];
let localTheta = null;
let localRound = -1;
let dataset = null;
let bytesUp = 0, bytesDown = 0;
let ws = null;
let wsConnected = false;
const uScratch = new Float32Array(P);
const thetaPlus = new Float32Array(P);
const thetaMinus = new Float32Array(P);
const thetaStep = new Float32Array(P);

function log(s) {
  const stamp = new Date().toLocaleTimeString();
  logEl.textContent = `[${stamp}] ${s}\n` + logEl.textContent.slice(0, 4000);
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

// ─── bootstrap ─

async function bootstrap() {
  if (!dataset) {
    const t0 = performance.now();
    dataset = await loadDataset();
    bytesDown += dataset.bytes;
    updateBwStat();
    log(`features loaded · ${dataset.N} examples · ${(dataset.bytes / 1024).toFixed(1)} KB · ${(performance.now() - t0).toFixed(0)}ms`);
  }
  const t0 = performance.now();
  const meta = await trackedFetch("/api/head-spsa-adaptive/snapshot");
  if (typeof meta.epsilon === "number") EPSILON = meta.epsilon;
  if (typeof meta.eta === "number") ETA = meta.eta;
  const buffers = await Promise.all(meta.shards.map(async s => {
    const r = await fetch(s.url);
    if (!r.ok) throw new Error(`shard${s.shard} ${r.status}`);
    const buf = await r.arrayBuffer();
    bytesDown += buf.byteLength;
    return { shard: s.shard, headerSize: s.shard === 0 ? 8 : 0, buf };
  }));
  updateBwStat();
  buffers.sort((a, b) => a.shard - b.shard);
  const headView = new DataView(buffers[0].buf);
  localRound = headView.getUint32(0, true);
  const p = headView.getUint32(4, true);
  if (p !== P) throw new Error(`P mismatch: server=${p} client=${P}`);
  localTheta = new Float32Array(P);
  for (const b of buffers) {
    const start = b.shard * (meta.shard_size_floats || (b.buf.byteLength - b.headerSize) / 4);
    localTheta.set(new Float32Array(b.buf, b.headerSize), start);
  }
  const ms = (performance.now() - t0).toFixed(0);
  log(`bootstrap @ R${localRound} · ${buffers.length} shards · ε=${EPSILON.toFixed(4)} η=${ETA.toExponential(2)} · ${ms}ms`);
}

// ─── network glue ─

async function tickPoll() {
  return await trackedFetch("/api/head-spsa-adaptive/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ worker_id: workerId, since_round: localRound }),
  });
}
async function submitBest(round, seed, scalar_g, delta) {
  return await trackedFetch("/api/head-spsa-adaptive/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      worker_id: workerId,
      round, seed, scalar_g, delta,
      since_round: localRound,
    }),
  });
}
function applyDelta(applied) {
  reconstructPerturbation(applied.seed, uScratch);
  const k = ETA * applied.scalar_g;
  for (let i = 0; i < P; i++) localTheta[i] -= k * uScratch[i];
}
async function reconcile(resp) {
  if (typeof resp.epsilon === "number") EPSILON = resp.epsilon;
  if (typeof resp.eta === "number") ETA = resp.eta;
  if (resp.round === localRound) return;
  if (typeof resp.oldest_applied_round === "number"
      && resp.oldest_applied_round > localRound + 1) {
    log("history truncated — re-bootstrap");
    await bootstrap();
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
  if (typeof s.last_loss === "number" && s.last_loss >= 0) lossEl.textContent = s.last_loss.toFixed(4);
  if (typeof s.last_acc  === "number" && s.last_acc  >= 0) accEl.textContent  = (s.last_acc * 100).toFixed(1) + "%";
  if (typeof s.eta === "number") etaEl.textContent = s.eta.toExponential(2);
  if (typeof s.accept_rate === "number") arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
}

// ─── WS push (best-effort; falls back to polling) ─

function openWebSocket() {
  try {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/api/head-spsa-adaptive/ws`);
    ws.addEventListener("open", () => { wsConnected = true; log("ws connected"); });
    ws.addEventListener("message", (e) => {
      bytesDown += e.data.length;
      updateBwStat();
      try {
        const m = JSON.parse(e.data);
        if (m.type === "advance" && localTheta) {
          if (m.applied && m.applied.round === localRound) applyDelta(m.applied);
          localRound = m.round;
        } else if (m.type === "hello" && localTheta) {
          if (typeof m.epsilon === "number") EPSILON = m.epsilon;
          if (typeof m.eta === "number") ETA = m.eta;
          if (Array.isArray(m.recent)) {
            for (const flip of m.recent) {
              if (flip.round >= localRound) applyDelta(flip);
            }
            localRound = m.round;
          }
        }
      } catch {}
    });
    ws.addEventListener("close", () => { wsConnected = false; log("ws closed — back to polling"); });
    ws.addEventListener("error", () => { wsConnected = false; });
  } catch (e) {
    log(`ws failed: ${e.message}`);
  }
}

// ─── chart (loss + accuracy over rounds) ─

function drawChart() {
  const w = chartCanvas.width, h = chartCanvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;
  const losses = history.map(p => p.loss).filter(v => v >= 0);
  const accs = history.map(p => p.acc).filter(v => v >= 0);
  const padX = 40, padY = 20;
  ctx.strokeStyle = "#d4d2cc"; ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, w - padX - 14, h - padY * 2);
  // loss (orange)
  if (losses.length >= 2) {
    const lo = Math.min(...losses), hi = Math.max(...losses);
    const range = Math.max(hi - lo, 0.001);
    ctx.strokeStyle = "#d6502c"; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      if (!(history[i].loss >= 0)) continue;
      const x = padX + (i / Math.max(history.length - 1, 1)) * (w - padX - 14);
      const y = (h - padY) - ((history[i].loss - lo) / range) * (h - padY * 2);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#666"; ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`loss ${hi.toFixed(3)}`, 4, padY + 8);
    ctx.fillText(`     ${lo.toFixed(3)}`, 4, h - padY);
  }
  // acc (purple, on the right axis)
  if (accs.length >= 2) {
    ctx.strokeStyle = "#6a4cb0"; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      if (!(history[i].acc >= 0)) continue;
      const x = padX + (i / Math.max(history.length - 1, 1)) * (w - padX - 14);
      const y = (h - padY) - history[i].acc * (h - padY * 2);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#6a4cb0";
    ctx.fillText(`acc 100%`, w - 60, padY + 8);
    ctx.fillText(`    0%`,   w - 60, h - padY);
  }
}

async function pullState() {
  try {
    const r = await fetch("/api/head-spsa-adaptive/state");
    const s = await r.json();
    history = (s.history || []).filter(h => h.round >= 0);
    drawChart();
    if (!running) {
      roundEl.textContent = s.round;
      poolEl.textContent = `${s.proposals}/${s.target}`;
      peersEl.textContent = (s.joined || []).length;
      if (typeof s.last_loss === "number" && s.last_loss >= 0) lossEl.textContent = s.last_loss.toFixed(4);
      if (typeof s.last_acc === "number" && s.last_acc >= 0) accEl.textContent = (s.last_acc * 100).toFixed(1) + "%";
      if (typeof s.eta === "number") etaEl.textContent = s.eta.toExponential(2);
      arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
    }
  } catch (e) {}
}

// ─── one SPSA trial — 3 forwards on this worker's shard ─

function spsaTrial(rng, lossBefore) {
  const seed = (rng() * 0xFFFFFFFF) >>> 0;
  reconstructPerturbation(seed, uScratch);
  for (let i = 0; i < P; i++) {
    thetaPlus[i] = localTheta[i] + EPSILON * uScratch[i];
    thetaMinus[i] = localTheta[i] - EPSILON * uScratch[i];
  }
  const lossPlus = classLoss(thetaPlus, dataset.features, dataset.labels, myShard.start, myShard.end);
  const lossMinus = classLoss(thetaMinus, dataset.features, dataset.labels, myShard.start, myShard.end);
  const scalar_g = (lossPlus - lossMinus) / (2 * EPSILON);
  const k = ETA * scalar_g;
  for (let i = 0; i < P; i++) thetaStep[i] = localTheta[i] - k * uScratch[i];
  const lossAt = classLoss(thetaStep, dataset.features, dataset.labels, myShard.start, myShard.end);
  return { seed, scalar_g, delta: lossAt - lossBefore };
}

async function runForever() {
  await bootstrap();
  openWebSocket();
  while (running) {
    try {
      if (!wsConnected) {
        const pulled = await tickPoll();
        updateStats(pulled);
        await reconcile(pulled);
      }
      const seedBase = ((localRound + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
      const rng = mulberry32(seedBase);
      let best;
      if (readAttackFlag()) {
        // Byzantine: send a random seed with a huge fake claimed_Δ.
        // Server's post-apply real_Δ check (Phase 39) will catch it.
        const fakeSeed = (rng() * 0xFFFFFFFF) >>> 0;
        best = { seed: fakeSeed, scalar_g: 1.0, delta: -10 };
      } else {
        const lossBefore = classLoss(localTheta, dataset.features, dataset.labels, myShard.start, myShard.end);
        best = null;
        for (let t = 0; t < TRIALS_PER_REPORT; t++) {
          const trial = spsaTrial(rng, lossBefore);
          if (!best || trial.delta < best.delta) best = trial;
        }
      }
      const reported = await submitBest(localRound, best.seed, best.scalar_g, best.delta);
      updateStats(reported);
      await reconcile(reported);
      if (reported.quarantined) {
        log(`⚠ QUARANTINED · proposals dropped — defense detected fraud`);
      } else if (reported.advanced) {
        const tag = readAttackFlag() ? " [attacker]" : "";
        log(`R${localRound - 1} → R${localRound} · Δ ${best.delta.toFixed(4)} · η ${ETA.toExponential(2)}${tag}`);
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
  log(`joining as ${workerId} → shard ${myShard.idx}`);
  pullState();
  runForever();
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset head-classifier coordinator? All progress lost.")) return;
  await fetch("/api/head-spsa-adaptive/reset", { method: "POST" });
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
