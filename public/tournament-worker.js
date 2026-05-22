/**
 * Browser worker — Phase 2 tournament protocol.
 *
 * Key difference from Phase 1: the coord no longer ships full θ on every
 * tick. Each worker bootstraps θ once via /api/tournament/snapshot, then
 * applies the accepted flip carried in each tick response to keep its
 * localTheta in sync. Downlink per round drops from ~520 B to ~36 B.
 *
 * For each round:
 *   1. Poll /api/tournament/tick (just to sync state — no proposal yet)
 *   2. Reconcile: apply last_applied if we trail by exactly 1 round;
 *      re-bootstrap if we trail by more
 *   3. Generate K=8 random flip proposals locally on localTheta
 *   4. Submit best proposal
 *   5. Reconcile again with the response
 */

const H = 32;
const P = 4 * H + 1;
const BATCH_SIZE = 128;
const TRIALS_PER_REPORT = 8;
const FLIP_SIZE = 4;
const FLIP_SIGMA = 0.15;
const POLL_DELAY_MS = 60;

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

function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function trueLabel(task, x, y) {
  if (task === "circle") return (x * x + y * y) < 1 ? 1 : 0;
  if (task === "xor")    return ((x > 0) !== (y > 0)) ? 1 : 0;
  return Math.sin(2 * x) > y ? 1 : 0;
}

function makeBatch(seed, n, task) {
  const rng = mulberry32(seed);
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = (rng() - 0.5) * 4;
    ys[i] = (rng() - 0.5) * 4;
    labels[i] = trueLabel(task, xs[i], ys[i]);
  }
  return { xs, ys, labels };
}

function forward(theta, x, y) {
  let z = theta[4 * H];
  const w2Off = 3 * H;
  const b1Off = 2 * H;
  for (let i = 0; i < H; i++) {
    let h = theta[i * 2] * x + theta[i * 2 + 1] * y + theta[b1Off + i];
    if (h < 0) h = 0;
    z += theta[w2Off + i] * h;
  }
  return 1 / (1 + Math.exp(-z));
}

function batchLoss(theta, batch) {
  let loss = 0;
  const N = batch.xs.length;
  const eps = 1e-7;
  for (let i = 0; i < N; i++) {
    const p = forward(theta, batch.xs[i], batch.ys[i]);
    const y = batch.labels[i];
    loss += -(y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps));
  }
  return loss / N;
}

function proposeFlip(theta, rng) {
  const indices = new Int32Array(FLIP_SIZE);
  const values = new Float32Array(FLIP_SIZE);
  const seen = new Set();
  for (let k = 0; k < FLIP_SIZE; k++) {
    let idx;
    do { idx = Math.floor(rng() * P); } while (seen.has(idx));
    seen.add(idx);
    indices[k] = idx;
    values[k] = theta[idx] + gaussian(rng) * FLIP_SIGMA;
  }
  return { indices, values };
}

// --- UI bindings ---
const workerId = `t-${Math.random().toString(36).slice(2, 8)}`;
const $ = (id) => document.getElementById(id);
const widEl = $("wid"), roundEl = $("round"), poolEl = $("pool");
const peersEl = $("peers"), lossEl = $("loss"), arEl = $("ar");
const bwEl = $("bw"), logEl = $("log");
const joinBtn = $("join"), resetBtn = $("reset");
const taskSel = $("task");
const chartCanvas = $("chart");
const ctx = chartCanvas.getContext("2d");
const boundaryCanvas = $("boundary");
const bctx = boundaryCanvas.getContext("2d");
const BG = boundaryCanvas.width;
const boundaryImage = bctx.createImageData(BG, BG);
widEl.textContent = workerId;

let running = false;
let history = [];
let currentTask = "wave";

// Phase 2 state
let localTheta = null;
let localRound = -1;
let bytesUp = 0;
let bytesDown = 0;
let bootstrapCount = 0;

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
  if (bwEl) {
    bwEl.textContent = `↑${fmtBytes(bytesUp)} ↓${fmtBytes(bytesDown)}`;
  }
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

async function bootstrap() {
  // Phase 3: snapshot is a binary blob served from R2 (or in-memory fallback).
  // Wire format: 4-byte LE uint32 round, 4-byte LE uint32 P, then P × float32 (LE).
  const meta = await trackedFetch("/api/tournament/snapshot");
  const r = await fetch(meta.snapshot_url);
  if (!r.ok) throw new Error(`snapshot.bin ${r.status}`);
  const buf = await r.arrayBuffer();
  bytesDown += buf.byteLength;
  updateBwStat();
  const view = new DataView(buf);
  const round = view.getUint32(0, true);
  const p = view.getUint32(4, true);
  if (p !== P) throw new Error(`P mismatch: server=${p} client=${P}`);
  localTheta = new Float32Array(buf, 8).slice();
  localRound = round;
  currentTask = meta.task || "wave";
  bootstrapCount += 1;
  log(`bootstrap @ R${round} via ${meta.snapshot_url} (${buf.byteLength} B binary)`);
  return meta;
}

async function tickPoll() {
  return await trackedFetch("/api/tournament/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ worker_id: workerId, since_round: localRound }),
  });
}

async function submitBest(round, indices, values, delta) {
  return await trackedFetch("/api/tournament/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      worker_id: workerId,
      round,
      indices: Array.from(indices),
      values: Array.from(values),
      delta,
      since_round: localRound,
    }),
  });
}

function applyDelta(applied) {
  for (let k = 0; k < applied.indices.length; k++) {
    const idx = applied.indices[k];
    if (idx >= 0 && idx < P) localTheta[idx] = applied.values[k];
  }
}

async function reconcile(resp) {
  currentTask = resp.task || currentTask;
  if (taskSel && taskSel.value !== currentTask) taskSel.value = currentTask;
  if (resp.round === localRound) return;
  // If server has truncated history past our last sync point, we must re-snapshot.
  if (typeof resp.oldest_applied_round === "number"
      && resp.oldest_applied_round > localRound + 1) {
    log(`history truncated (oldest=${resp.oldest_applied_round}, local=${localRound}) — re-bootstrapping`);
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
  lossEl.textContent = s.last_loss >= 0 ? s.last_loss.toFixed(4) : "—";
  if (typeof s.accept_rate === "number") {
    arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
  }
}

function drawChart() {
  const w = chartCanvas.width, h = chartCanvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;
  const losses = history.map(p => p.loss);
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const range = Math.max(hi - lo, 0.001);
  const padX = 36, padY = 18;
  ctx.strokeStyle = "#d4d2cc";
  ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, w - padX - 14, h - padY * 2);
  ctx.strokeStyle = "#d6502c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = padX + (i / Math.max(history.length - 1, 1)) * (w - padX - 14);
    const y = (h - padY) - ((history[i].loss - lo) / range) * (h - padY * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#2c8a4f";
  for (let i = 0; i < history.length; i++) {
    if (history[i].accepted) {
      const x = padX + (i / Math.max(history.length - 1, 1)) * (w - padX - 14);
      ctx.fillRect(x - 1, h - padY + 2, 2, 4);
    }
  }
  ctx.fillStyle = "#666";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(hi.toFixed(3), 4, padY + 8);
  ctx.fillText(lo.toFixed(3), 4, h - padY + 4);
  ctx.fillText(`R${history[0].round} → R${history[history.length - 1].round}`, padX + 4, h - 4);
}

function renderBoundary(theta) {
  const data = boundaryImage.data;
  for (let py = 0; py < BG; py++) {
    const y = 2 - (py / (BG - 1)) * 4;
    for (let px = 0; px < BG; px++) {
      const x = -2 + (px / (BG - 1)) * 4;
      const p = forward(theta, x, y);
      const r = Math.round(60 + (220 - 60) * p);
      const g = Math.round(110 + (90 - 110) * p);
      const b = Math.round(200 + (50 - 200) * p);
      const idx = (py * BG + px) * 4;
      data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = 255;
    }
  }
  bctx.putImageData(boundaryImage, 0, 0);
  bctx.strokeStyle = "rgba(20,20,20,0.85)";
  bctx.lineWidth = 1.5;
  bctx.beginPath();
  const toPx = (x, y) => [((x + 2) / 4) * (BG - 1), ((2 - y) / 4) * (BG - 1)];
  if (currentTask === "wave") {
    for (let px = 0; px < BG; px++) {
      const x = -2 + (px / (BG - 1)) * 4;
      const [, py] = toPx(x, Math.sin(2 * x));
      if (px === 0) bctx.moveTo(px, py);
      else bctx.lineTo(px, py);
    }
  } else if (currentTask === "circle") {
    bctx.arc(BG / 2, BG / 2, (1 / 4) * (BG - 1), 0, Math.PI * 2);
  } else if (currentTask === "xor") {
    const [cx, cy] = toPx(0, 0);
    bctx.moveTo(0, cy); bctx.lineTo(BG, cy);
    bctx.moveTo(cx, 0); bctx.lineTo(cx, BG);
  }
  bctx.stroke();
  bctx.strokeStyle = "rgba(0,0,0,0.15)";
  bctx.strokeRect(0, 0, BG, BG);
}

async function pullState() {
  // Lightweight summary fetch for the chart + boundary preview when idle.
  // Still includes theta (only ~520 B and called every 5s).
  try {
    const r = await fetch("/api/tournament/state");
    const s = await r.json();
    history = s.history.filter(h => h.round >= 0);
    currentTask = s.task || "wave";
    if (taskSel && taskSel.value !== currentTask) taskSel.value = currentTask;
    drawChart();
    if (!running && s.theta) renderBoundary(new Float32Array(s.theta));
    if (running && localTheta) renderBoundary(localTheta);
    if (!running) {
      roundEl.textContent = s.round;
      poolEl.textContent = `${s.proposals}/${s.target}`;
      peersEl.textContent = (s.joined || []).length;
      lossEl.textContent = s.last_loss >= 0 ? s.last_loss.toFixed(4) : "—";
      arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
    }
  } catch (e) {}
}

async function runForever() {
  await bootstrap();
  while (running) {
    try {
      // Pull state for this round (no proposal — just sync)
      const pulled = await tickPoll();
      updateStats(pulled);
      await reconcile(pulled);
      renderBoundary(localTheta);

      // Generate K proposals on localTheta
      const seed = ((localRound + 1) * 1000003) ^
                   (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
      const rng = mulberry32(seed);
      const batch = makeBatch(seed ^ 0xA17BEEF, BATCH_SIZE, currentTask);
      const lossBefore = batchLoss(localTheta, batch);

      const trial = new Float32Array(P);
      let best = null;
      for (let t = 0; t < TRIALS_PER_REPORT; t++) {
        const { indices, values } = proposeFlip(localTheta, rng);
        for (let i = 0; i < P; i++) trial[i] = localTheta[i];
        for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
        const lossAfter = batchLoss(trial, batch);
        const delta = lossAfter - lossBefore;
        if (!best || delta < best.delta) {
          best = { indices, values, delta };
        }
      }

      const reported = await submitBest(localRound, best.indices, best.values, best.delta);
      updateStats(reported);
      await reconcile(reported);
      if (reported.advanced) {
        log(`R${localRound - 1} → R${localRound} · best Δ ${best.delta.toFixed(4)} · loss ${reported.last_loss.toFixed(4)} · ↑${fmtBytes(bytesUp)} ↓${fmtBytes(bytesDown)}`);
      } else if (reported.rejected) {
        log(`stale @ local R${localRound} — coord at R${reported.round}`);
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
  log(`Joining as ${workerId}`);
  pullState();
  runForever();
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset tournament coordinator? All progress lost.")) return;
  await fetch("/api/tournament/reset", { method: "POST" });
  history = [];
  localTheta = null;
  localRound = -1;
  bytesUp = 0;
  bytesDown = 0;
  updateBwStat();
  drawChart();
  log("reset");
});

if (taskSel) {
  taskSel.addEventListener("change", async (e) => {
    const newTask = e.target.value;
    if (!confirm(`Switch task to "${newTask}"? Coord resets, all progress lost.`)) {
      e.target.value = currentTask;
      return;
    }
    await fetch("/api/tournament/set_task", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task: newTask }),
    });
    history = [];
    localTheta = null;
    localRound = -1;
    bytesUp = 0;
    bytesDown = 0;
    updateBwStat();
    currentTask = newTask;
    log(`task → ${newTask}`);
    await pullState();
  });
}

pullState();
setInterval(pullState, 5000);
