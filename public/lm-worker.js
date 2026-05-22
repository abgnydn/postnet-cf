/**
 * Browser worker — Phase 5 char-LM tournament.
 * Float-weight bigram model: embed (V×E) + linear (E×V) + bias (V).
 * Same flip-and-accept protocol as Phase 2/3 Tournament.
 */

const V = 27, E = 16;
const P_EMBED = V * E;
const P_OUT = E * V;
const P_BIAS = V;
const P = P_EMBED + P_OUT + P_BIAS;   // 891
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

function forward(theta, charIdx, logits) {
  const eStart = charIdx * E;
  for (let v = 0; v < V; v++) logits[v] = theta[P_EMBED + P_OUT + v];
  for (let i = 0; i < E; i++) {
    const ei = theta[eStart + i];
    for (let v = 0; v < V; v++) {
      logits[v] += ei * theta[P_EMBED + i * V + v];
    }
  }
}

function textLoss(theta) {
  const logits = new Float32Array(V);
  let loss = 0;
  for (let i = 0; i < CODES.length - 1; i++) {
    forward(theta, CODES[i], logits);
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[v] - mx);
    const target = CODES[i + 1];
    loss += -(logits[target] - mx - Math.log(sum + 1e-7));
  }
  return loss / (CODES.length - 1);
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
  let c = charCode("t");
  const out = [c];
  for (let k = 0; k < n; k++) {
    forward(theta, c, logits);
    let mx = -Infinity, arg = 0;
    for (let v = 0; v < V; v++) {
      const score = logits[v] + (rng() - 0.5) * 0.3;
      if (score > mx) { mx = score; arg = v; }
    }
    c = arg;
    out.push(c);
  }
  return out.map(k => k === 0 ? ' ' : String.fromCharCode(97 + k - 1)).join('');
}

// --- UI bindings ---
const workerId = `lm-${Math.random().toString(36).slice(2, 8)}`;
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

async function bootstrap() {
  // Phase 6: snapshot is sharded. Manifest lists N shards; we fetch all in parallel.
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
  // First shard carries header; the rest are raw floats
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

async function runForever() {
  await bootstrap();
  const trial = new Float32Array(P);
  while (running) {
    try {
      const pulled = await tickPoll();
      updateStats(pulled);
      await reconcile(pulled);

      const seed = ((localRound + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
      const rng = mulberry32(seed);
      const lossBefore = textLoss(localTheta);
      let best = null;
      for (let t = 0; t < TRIALS_PER_REPORT; t++) {
        const { indices, values } = proposeFlip(localTheta, rng);
        for (let i = 0; i < P; i++) trial[i] = localTheta[i];
        for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
        const lossAfter = textLoss(trial);
        const delta = lossAfter - lossBefore;
        if (!best || delta < best.delta) best = { indices, values, delta };
      }

      const reported = await submitBest(localRound, best.indices, best.values, best.delta);
      updateStats(reported);
      await reconcile(reported);
      if (reported.advanced) {
        log(`R${localRound - 1} → R${localRound} · Δ ${best.delta.toFixed(4)} · loss ${reported.last_loss.toFixed(4)} · ↑${fmtBytes(bytesUp)} ↓${fmtBytes(bytesDown)}`);
        if (localRound % 25 === 0) updateSample();
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
  log(`Joining as ${workerId}`);
  pullState();
  runForever();
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset LM coordinator? All progress lost.")) return;
  await fetch("/api/lm/reset", { method: "POST" });
  history = [];
  localTheta = null;
  localRound = -1;
  bytesUp = 0; bytesDown = 0;
  updateBwStat();
  drawChart();
  if (sampleEl) sampleEl.textContent = "(idle)";
  log("reset");
});

pullState();
setInterval(pullState, 5000);
