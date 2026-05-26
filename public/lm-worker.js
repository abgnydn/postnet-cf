/**
 * Browser worker — Phase 5 char-LM tournament.
 * Float-weight bigram model: embed (V×E) + linear (E×V) + bias (V).
 * Same flip-and-accept protocol as Phase 2/3 Tournament.
 */

// Phase 8: context-2 MLP. P = 2379. Must match server src/tournament-lm.ts
const V = 27, E = 16, HID = 32, CTX = 2;
const P_EMBED = V * E;                  // 432
const P_FC1 = CTX * E * HID;            // 1024
const P_B1 = HID;                       // 32
const P_FC2 = HID * V;                  // 864
const P_B2 = V;                         // 27
const P = P_EMBED + P_FC1 + P_B1 + P_FC2 + P_B2;   // 2379
const FC1_OFF = P_EMBED;
const B1_OFF = FC1_OFF + P_FC1;
const FC2_OFF = B1_OFF + P_B1;
const B2_OFF = FC2_OFF + P_FC2;
const TRIALS_PER_REPORT = 8;
const FLIP_SIZE = 6;
const FLIP_SIGMA = 0.15;
const POLL_DELAY_MS = 60;

const TEXT =
  "the bird sings every dawn the cat sleeps in the sun the dog runs to the park " +
  "the wind blows the leaves the rain falls in the night the moon shines bright " +
  "above the mountains rise high and the river flows fast through the woods and " +
  "over the stones to the wide open sea where the waves break and roll back " +
  "again and again forever and ever";

function charCode(c) {
  if (c === ' ') return 0;
  const k = c.charCodeAt(0) - 97;
  return k >= 0 && k < 26 ? k + 1 : 0;
}
const CODES = new Uint8Array(TEXT.length);
for (let i = 0; i < TEXT.length; i++) CODES[i] = charCode(TEXT[i]);

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

function forward(theta, prevPrev, prev, logits) {
  // x = concat(embed[prevPrev], embed[prev])  (2E)
  const x = new Float32Array(CTX * E);
  for (let i = 0; i < E; i++) {
    x[i] = theta[prevPrev * E + i];
    x[E + i] = theta[prev * E + i];
  }
  const h = new Float32Array(HID);
  for (let j = 0; j < HID; j++) h[j] = theta[B1_OFF + j];
  for (let i = 0; i < CTX * E; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = FC1_OFF + i * HID;
    for (let j = 0; j < HID; j++) h[j] += xi * theta[row + j];
  }
  for (let j = 0; j < HID; j++) if (h[j] < 0) h[j] = 0;
  for (let v = 0; v < V; v++) logits[v] = theta[B2_OFF + v];
  for (let j = 0; j < HID; j++) {
    const hj = h[j];
    if (hj === 0) continue;
    const row = FC2_OFF + j * V;
    for (let v = 0; v < V; v++) logits[v] += hj * theta[row + v];
  }
}

// Phase 7: federated data shards. Each worker hashes its workerId to pick
// a deterministic disjoint slice of TEXT. The coord still reports loss on
// the full text as the convergence metric, but proposal scoring uses only
// the worker's private shard. Workers with different data can disagree
// about which flip is best — that's the "federated" in federated learning.
const NUM_SHARDS = 3;
function shardForWorker(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  const s = h % NUM_SHARDS;
  const span = Math.floor(CODES.length / NUM_SHARDS);
  return { start: s * span, end: s === NUM_SHARDS - 1 ? CODES.length : (s + 1) * span, idx: s };
}

function textLoss(theta, shardStart, shardEnd) {
  const logits = new Float32Array(V);
  let loss = 0;
  // Need CTX previous chars before predicting; clamp start
  const start = Math.max(shardStart != null ? shardStart : 0, CTX);
  const end = shardEnd != null ? shardEnd : CODES.length;
  for (let i = start; i < end; i++) {
    forward(theta, CODES[i - 2], CODES[i - 1], logits);
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[v] - mx);
    const target = CODES[i];
    loss += -(logits[target] - mx - Math.log(sum + 1e-7));
  }
  return loss / Math.max(end - start, 1);
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

function sampleText(theta, seed, n) {
  const rng = mulberry32(seed);
  const logits = new Float32Array(V);
  let prevPrev = 0;
  let prev = charCode("t");
  const out = [prev];
  for (let k = 0; k < n; k++) {
    forward(theta, prevPrev, prev, logits);
    let mx = -Infinity, arg = 0;
    for (let v = 0; v < V; v++) {
      const score = logits[v] + (rng() - 0.5) * 0.3;
      if (score > mx) { mx = score; arg = v; }
    }
    prevPrev = prev;
    prev = arg;
    out.push(arg);
  }
  return out.map(k => k === 0 ? ' ' : String.fromCharCode(97 + k - 1)).join('');
}

// Phase 23: attack-mode toggle. Set via URL param ?attack=1 or via the
// checkbox in /lm.html. When true, this worker submits random flips with
// claimed delta=-10 (the standard byzantine pattern).
function readAttackFlag() {
  if (typeof location !== "undefined") {
    if (new URLSearchParams(location.search).get("attack") === "1") return true;
  }
  const el = typeof document !== "undefined" ? document.getElementById("attack") : null;
  return !!(el && el.checked);
}

// --- UI bindings ---
const workerId = `lm-${Math.random().toString(36).slice(2, 8)}`;
const myShard = shardForWorker(workerId);
const $ = (id) => document.getElementById(id);
const widEl = $("wid"), roundEl = $("round"), poolEl = $("pool");
const peersEl = $("peers"), lossEl = $("loss"), arEl = $("ar");
const bwEl = $("bw"), logEl = $("log"), sampleEl = $("sample");
const joinBtn = $("join"), resetBtn = $("reset");
const chartCanvas = $("chart");
const ctx = chartCanvas.getContext("2d");
widEl.textContent = workerId;

let running = false;
let history = [];
let localTheta = null;
let localRound = -1;
let bytesUp = 0, bytesDown = 0;
// Phase 10: WebSocket push
let ws = null;
let wsConnected = false;

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

// Phase 33: localStorage cache for theta. Revisits skip the full snapshot
// fetch and resume from the cached round; coord ships deltas via applied_since.
const CACHE_KEY = "postnet.lm.theta.v1";

function loadCachedTheta() {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const { round, theta } = JSON.parse(raw);
    if (typeof round !== "number" || !Array.isArray(theta) || theta.length !== P) return null;
    return { round, theta: new Float32Array(theta) };
  } catch { return null; }
}
function saveCachedTheta(round, theta) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify({ round, theta: Array.from(theta) }));
  } catch {}
}

async function bootstrap() {
  // Phase 33: try cache first
  const cached = loadCachedTheta();
  if (cached) {
    localTheta = cached.theta;
    localRound = cached.round;
    log(`bootstrap from localStorage @ R${cached.round} (no network)`);
    return null;
  }
  // Phase 6: full sharded fetch
  const t0 = performance.now();
  const meta = await trackedFetch("/api/lm/snapshot");
  if (!Array.isArray(meta.shards)) throw new Error("no shards in manifest");
  const buffers = await Promise.all(meta.shards.map(async s => {
    const r = await fetch(s.url);
    if (!r.ok) throw new Error(`shard${s.shard} ${r.status}`);
    const buf = await r.arrayBuffer();
    bytesDown += buf.byteLength;
    return { shard: s.shard, headerSize: s.shard === 0 ? 8 : 0, buf };
  }));
  updateBwStat();
  buffers.sort((a, b) => a.shard - b.shard);
  const head = buffers[0];
  const headView = new DataView(head.buf);
  const round = headView.getUint32(0, true);
  const p = headView.getUint32(4, true);
  if (p !== P) throw new Error(`P mismatch: server=${p} client=${P}`);
  localTheta = new Float32Array(P);
  for (const b of buffers) {
    const start = b.shard * (meta.shard_size_floats || (b.buf.byteLength - b.headerSize) / 4);
    const floats = new Float32Array(b.buf, b.headerSize);
    localTheta.set(floats, start);
  }
  localRound = round;
  saveCachedTheta(round, localTheta);
  const ms = (performance.now() - t0).toFixed(0);
  const totalBytes = buffers.reduce((a, b) => a + b.buf.byteLength, 0);
  log(`bootstrap @ R${round} via ${buffers.length} shards · ${totalBytes} B · ${ms}ms`);
  return meta;
}

async function tickPoll() {
  return await trackedFetch("/api/lm/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ worker_id: workerId, since_round: localRound }),
  });
}
async function submitBest(round, indices, values, delta) {
  return await trackedFetch("/api/lm/tick", {
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
    localTheta[applied.indices[k]] = applied.values[k];
  }
}
async function reconcile(resp) {
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
  lossEl.textContent = s.last_loss >= 0 ? s.last_loss.toFixed(4) : "—";
  if (typeof s.accept_rate === "number") arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
}
function drawChart() {
  const w = chartCanvas.width, h = chartCanvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;
  const losses = history.map(p => p.loss);
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const range = Math.max(hi - lo, 0.001);
  const padX = 36, padY = 18;
  ctx.strokeStyle = "#d4d2cc"; ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, w - padX - 14, h - padY * 2);
  ctx.strokeStyle = "#d6502c"; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = padX + (i / Math.max(history.length - 1, 1)) * (w - padX - 14);
    const y = (h - padY) - ((history[i].loss - lo) / range) * (h - padY * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#666";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(hi.toFixed(3), 4, padY + 8);
  ctx.fillText(lo.toFixed(3), 4, h - padY + 4);
  ctx.fillText(`log(V)=${Math.log(V).toFixed(2)} (chance)`, padX + 4, h - 4);
}
function updateSample() {
  if (localTheta && sampleEl) {
    const s = sampleText(localTheta, Date.now() & 0xFFFFFFFF, 100);
    sampleEl.textContent = s;
  }
}

async function pullState() {
  try {
    const r = await fetch("/api/lm/state");
    const s = await r.json();
    history = s.history.filter(h => h.round >= 0);
    drawChart();
    if (!running) {
      roundEl.textContent = s.round;
      poolEl.textContent = `${s.proposals}/${s.target}`;
      peersEl.textContent = (s.joined || []).length;
      lossEl.textContent = s.last_loss >= 0 ? s.last_loss.toFixed(4) : "—";
      arEl.textContent = (s.accept_rate * 100).toFixed(1) + "%";
      if (s.sample && sampleEl) sampleEl.textContent = s.sample;
    } else {
      updateSample();
    }
  } catch (e) {}
}

function openWebSocket() {
  try {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/api/lm/ws`);
    ws.addEventListener("open", () => {
      wsConnected = true;
      log("ws connected");
    });
    ws.addEventListener("message", (e) => {
      bytesDown += e.data.length;
      updateBwStat();
      try {
        const m = JSON.parse(e.data);
        if (m.type === "advance" && localTheta) {
          if (m.applied && m.applied.round === localRound) applyDelta(m.applied);
          localRound = m.round;
          if (localRound % 25 === 0) updateSample();
          renderBoundaryNoop();
        } else if (m.type === "hello" && localTheta && Array.isArray(m.recent)) {
          for (const flip of m.recent) {
            if (flip.round >= localRound) applyDelta(flip);
          }
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

function renderBoundaryNoop() {}  // hook for future renderBoundary() integration

async function runForever() {
  await bootstrap();
  openWebSocket();
  const trial = new Float32Array(P);
  while (running) {
    try {
      // Phase 10: skip the explicit poll when WS is providing pushes
      if (!wsConnected) {
        const pulled = await tickPoll();
        updateStats(pulled);
        await reconcile(pulled);
      }

      const seed = ((localRound + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
      const rng = mulberry32(seed);
      let best;
      if (readAttackFlag()) {
        // Phase 23: byzantine — random flip, claimed huge improvement
        const { indices, values } = proposeFlip(localTheta, rng);
        best = { indices, values, delta: -10 };
      } else {
        // Phase 7: score only on this worker's private text shard
        const lossBefore = textLoss(localTheta, myShard.start, myShard.end);
        best = null;
        for (let t = 0; t < TRIALS_PER_REPORT; t++) {
          const { indices, values } = proposeFlip(localTheta, rng);
          for (let i = 0; i < P; i++) trial[i] = localTheta[i];
          for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
          const lossAfter = textLoss(trial, myShard.start, myShard.end);
          const delta = lossAfter - lossBefore;
          if (!best || delta < best.delta) best = { indices, values, delta };
        }
      }

      const reported = await submitBest(localRound, best.indices, best.values, best.delta);
      updateStats(reported);
      await reconcile(reported);
      if (reported.quarantined) {
        log(`⚠ QUARANTINED · proposals dropped — defense detected fraud`);
      } else if (reported.advanced) {
        const mode = readAttackFlag() ? " [attacker]" : "";
        log(`R${localRound - 1} → R${localRound} · Δ ${best.delta.toFixed(4)} · loss ${reported.last_loss.toFixed(4)}${mode}`);
        if (localRound % 25 === 0) updateSample();
        // Phase 33: periodically refresh the cached theta
        if (localRound % 50 === 0) saveCachedTheta(localRound, localTheta);
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
  log(`Joining as ${workerId} · shard${myShard.idx} [${myShard.start}..${myShard.end}] of ${CODES.length}`);
  pullState();
  runForever();
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset LM coordinator? All progress lost.")) return;
  await fetch("/api/lm/reset", { method: "POST" });
  try { localStorage.removeItem(CACHE_KEY); } catch {}
  history = [];
  localTheta = null;
  localRound = -1;
  bytesUp = 0; bytesDown = 0;
  updateBwStat();
  drawChart();
  if (sampleEl) sampleEl.textContent = "(idle)";
  log("reset (cache cleared)");
});

pullState();
setInterval(pullState, 5000);
