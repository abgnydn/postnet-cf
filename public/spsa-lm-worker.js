/**
 * Browser worker — Phase 36 char-LM SPSA tournament.
 *
 * Same model and bootstrap as lm-worker.js. The propose-and-score loop is
 * replaced with SPSA:
 *
 *   for t in 1..T:
 *     seed_t  = random u32
 *     u       = Gaussian(seed_t)           (bit-reproducible on the server)
 *     g_t     = (textLoss(θ+εu) - textLoss(θ-εu)) / (2ε)
 *     lossAt  =  textLoss(θ - η·g_t·u)
 *     Δ_t     =  lossAt - lossBefore       (shard-local)
 *   pick best Δ_t, submit (seed, scalar_g, delta)
 *
 * The server reconstructs u from seed (mulberry32+Box-Muller, identical to
 * this file) and applies θ ← θ − η·scalar_g·u.
 *
 * Wire format per proposal: 20 bytes  (vs 52 for the random-flip variant).
 *
 * Acceptance for Phase 0 (see CLAUDE.md): converges on the same char-LM as
 * the flip-and-accept worker, within ≤ 2× rounds. Byzantine attack mode
 * still gets quarantined.
 */

const V = 27, E = 16, HID = 32, CTX = 2;
const P_EMBED = V * E;
const P_FC1 = CTX * E * HID;
const P_B1 = HID;
const P_FC2 = HID * V;
const P_B2 = V;
const P = P_EMBED + P_FC1 + P_B1 + P_FC2 + P_B2;   // 2379
const FC1_OFF = P_EMBED;
const B1_OFF = FC1_OFF + P_FC1;
const FC2_OFF = B1_OFF + P_B1;
const B2_OFF = FC2_OFF + P_FC2;

// SPSA hyperparams. Echo of the server-side constants. If the server's
// /tick response carries different values, we update at runtime.
let EPSILON = 0.01;
let ETA = 0.005;
const TRIALS_PER_REPORT = 4;
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

// CRITICAL: must be bit-identical to src/tournament-spsa-lm.ts. The server
// applies the worker's update by reconstructing u from this exact code.
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

function forward(theta, prevPrev, prev, logits) {
  const x = new Float32Array(CTX * E);
  for (let i = 0; i < E; i++) {
    x[i] = theta[prevPrev * E + i];
    x[E + i] = theta[prev * E + i];
  }
  const h = new Float32Array(HID);
  for (let j = 0; j < HID; j++) h[j] = theta[B1_OFF + j];
  for (let i = 0; i < CTX * E; i++) {
    const xi = x[i]; if (xi === 0) continue;
    const row = FC1_OFF + i * HID;
    for (let j = 0; j < HID; j++) h[j] += xi * theta[row + j];
  }
  for (let j = 0; j < HID; j++) if (h[j] < 0) h[j] = 0;
  for (let v = 0; v < V; v++) logits[v] = theta[B2_OFF + v];
  for (let j = 0; j < HID; j++) {
    const hj = h[j]; if (hj === 0) continue;
    const row = FC2_OFF + j * V;
    for (let v = 0; v < V; v++) logits[v] += hj * theta[row + v];
  }
}

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

function readAttackFlag() {
  if (typeof location !== "undefined") {
    if (new URLSearchParams(location.search).get("attack") === "1") return true;
  }
  const el = typeof document !== "undefined" ? document.getElementById("attack") : null;
  return !!(el && el.checked);
}

// --- UI bindings (paralleling lm-worker.js) ---
const workerId = `spsa-${Math.random().toString(36).slice(2, 8)}`;
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
let ws = null;
let wsConnected = false;
// Scratch buffers for the inner SPSA loop. Reused every trial.
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

const CACHE_KEY = "postnet.spsa-lm.theta.v1";
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
  const cached = loadCachedTheta();
  if (cached) {
    localTheta = cached.theta;
    localRound = cached.round;
    log(`bootstrap from localStorage @ R${cached.round} (no network)`);
    return null;
  }
  const t0 = performance.now();
  const meta = await trackedFetch("/api/spsa-lm/snapshot");
  if (!Array.isArray(meta.shards)) throw new Error("no shards in manifest");
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
  log(`bootstrap @ R${round} via ${buffers.length} shards · ${totalBytes} B · ${ms}ms · ε=${EPSILON} η=${ETA}`);
  return meta;
}

async function tickPoll() {
  return await trackedFetch("/api/spsa-lm/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ worker_id: workerId, since_round: localRound }),
  });
}
async function submitBest(round, seed, scalar_g, delta) {
  return await trackedFetch("/api/spsa-lm/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      worker_id: workerId,
      round,
      seed,
      scalar_g,
      delta,
      since_round: localRound,
    }),
  });
}
function applyDelta(applied) {
  // Reconstruct u from seed (bit-identical to server) and apply
  // θ ← θ − η·scalar_g·u. ETA may have been updated from /snapshot;
  // appliedHistory replay assumes ETA is stable for the run.
  reconstructPerturbation(applied.seed, uScratch);
  const k = ETA * applied.scalar_g;
  for (let i = 0; i < P; i++) localTheta[i] -= k * uScratch[i];
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
  if (typeof s.epsilon === "number") EPSILON = s.epsilon;
  if (typeof s.eta === "number") ETA = s.eta;
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
    const r = await fetch("/api/spsa-lm/state");
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
    ws = new WebSocket(`${proto}//${location.host}/api/spsa-lm/ws`);
    ws.addEventListener("open", () => { wsConnected = true; log("ws connected"); });
    ws.addEventListener("message", (e) => {
      bytesDown += e.data.length;
      updateBwStat();
      try {
        const m = JSON.parse(e.data);
        if (m.type === "advance" && localTheta) {
          if (m.applied && m.applied.round === localRound) applyDelta(m.applied);
          localRound = m.round;
          if (localRound % 25 === 0) updateSample();
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

// One SPSA trial: probe along a random Gaussian direction, return
// { seed, scalar_g, delta } (delta is local-shard, not global).
function spsaTrial(rng, lossBefore) {
  const seed = (rng() * 0xFFFFFFFF) >>> 0;
  reconstructPerturbation(seed, uScratch);
  for (let i = 0; i < P; i++) {
    thetaPlus[i] = localTheta[i] + EPSILON * uScratch[i];
    thetaMinus[i] = localTheta[i] - EPSILON * uScratch[i];
  }
  const lossPlus = textLoss(thetaPlus, myShard.start, myShard.end);
  const lossMinus = textLoss(thetaMinus, myShard.start, myShard.end);
  const scalar_g = (lossPlus - lossMinus) / (2 * EPSILON);
  const k = ETA * scalar_g;
  for (let i = 0; i < P; i++) thetaStep[i] = localTheta[i] - k * uScratch[i];
  const lossAt = textLoss(thetaStep, myShard.start, myShard.end);
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
        // Byzantine: fabricate a proposal with a random seed + a wildly
        // negative claimed delta. The server's post-apply real_Δ check
        // catches this exactly as for flip-and-accept — the fraud signal
        // is loss-going-up-after-apply, regardless of the update math.
        const fakeSeed = (rng() * 0xFFFFFFFF) >>> 0;
        best = { seed: fakeSeed, scalar_g: 1.0, delta: -10 };
      } else {
        const lossBefore = textLoss(localTheta, myShard.start, myShard.end);
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
        const mode = readAttackFlag() ? " [attacker]" : "";
        log(`R${localRound - 1} → R${localRound} · Δ ${best.delta.toFixed(4)} · g ${best.scalar_g.toFixed(3)} · loss ${reported.last_loss.toFixed(4)}${mode}`);
        if (localRound % 25 === 0) updateSample();
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
  log(`Joining as ${workerId} · shard${myShard.idx} [${myShard.start}..${myShard.end}] · SPSA ε=${EPSILON} η=${ETA}`);
  pullState();
  runForever();
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset SPSA-LM coordinator? All progress lost.")) return;
  await fetch("/api/spsa-lm/reset", { method: "POST" });
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
