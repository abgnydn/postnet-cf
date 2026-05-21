/**
 * Browser-side FedSGD worker.
 *
 * Pull current θ from the coord, compute ∇L on a local batch via manual
 * backprop, POST the gradient back. The coord averages all workers'
 * gradients in a round and steps θ.
 */

const H = 32;
const P = 4 * H + 1;       // 129 params — must match server
const BATCH_SIZE = 64;
const POLL_DELAY_MS = 80;

// --- PRNG for synthetic data ---
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

// --- Forward pass (same shape as server) ---
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

// --- Manual backprop. Returns ∇L over the batch, mean-reduced. ---
function computeGradient(theta, batch) {
  const grad = new Float32Array(P);
  const N = batch.xs.length;
  const w2Off = 3 * H;
  const b1Off = 2 * H;
  const b2Off = 4 * H;
  const pre = new Float32Array(H);
  const h = new Float32Array(H);

  for (let n = 0; n < N; n++) {
    const x = batch.xs[n];
    const y = batch.ys[n];
    const label = batch.labels[n];

    // Forward
    let z = theta[b2Off];
    for (let i = 0; i < H; i++) {
      const v = theta[i * 2] * x + theta[i * 2 + 1] * y + theta[b1Off + i];
      pre[i] = v;
      const hv = v > 0 ? v : 0;
      h[i] = hv;
      z += theta[w2Off + i] * hv;
    }
    const p = 1 / (1 + Math.exp(-z));

    // Backward (BCE + sigmoid → dL/dz = p − label)
    const dz = p - label;
    grad[b2Off] += dz;
    for (let i = 0; i < H; i++) {
      grad[w2Off + i] += dz * h[i];
      if (pre[i] > 0) {
        const dpre = dz * theta[w2Off + i];
        grad[i * 2] += dpre * x;
        grad[i * 2 + 1] += dpre * y;
        grad[b1Off + i] += dpre;
      }
    }
  }
  const invN = 1 / N;
  for (let i = 0; i < P; i++) grad[i] *= invN;
  return grad;
}

function trueLabel(task, x, y) {
  if (task === "circle") return (x * x + y * y) < 1 ? 1 : 0;
  if (task === "xor")    return ((x > 0) !== (y > 0)) ? 1 : 0;
  return Math.sin(2 * x) > y ? 1 : 0;  // "wave" default
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

// --- UI bindings ---
const workerId = `w-${Math.random().toString(36).slice(2, 8)}`;
const $ = (id) => document.getElementById(id);
const widEl = $("wid"), roundEl = $("round"), poolEl = $("pool");
const peersEl = $("peers"), lossEl = $("loss"), logEl = $("log");
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

function log(s) {
  const stamp = new Date().toLocaleTimeString();
  logEl.textContent = `[${stamp}] ${s}\n` + logEl.textContent.slice(0, 4000);
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
  // Ground-truth overlay per task
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

async function fetchTheta() {
  const r = await fetch("/api/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ worker_id: workerId }),
  });
  if (!r.ok) throw new Error(`coord ${r.status}`);
  return await r.json();
}

async function reportGradient(round, gradient) {
  const r = await fetch("/api/tick", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ worker_id: workerId, round, gradient }),
  });
  if (!r.ok) throw new Error(`coord ${r.status}`);
  return await r.json();
}

function updateStats(s) {
  roundEl.textContent = s.round;
  poolEl.textContent = `${s.pool_size}/${s.target}`;
  peersEl.textContent = s.joined;
  lossEl.textContent = s.last_loss >= 0 ? s.last_loss.toFixed(4) : "—";
}

async function pullHistory() {
  try {
    const r = await fetch("/api/state");
    const s = await r.json();
    history = s.history.filter(h => h.round >= 0);
    currentTask = s.task || "wave";
    if (taskSel && taskSel.value !== currentTask) taskSel.value = currentTask;
    drawChart();
    if (s.theta) renderBoundary(new Float32Array(s.theta));
    if (!running) {
      roundEl.textContent = s.round;
      poolEl.textContent = `${s.pool_size}/${s.target}`;
      peersEl.textContent = (s.joined || []).length;
      lossEl.textContent = s.last_loss >= 0 ? s.last_loss.toFixed(4) : "—";
    }
  } catch (e) { /* ignore */ }
}

async function runForever() {
  while (running) {
    try {
      const pulled = await fetchTheta();
      updateStats(pulled);
      currentTask = pulled.task || "wave";
      if (taskSel && taskSel.value !== currentTask) taskSel.value = currentTask;
      const theta = new Float32Array(pulled.theta);
      const myRound = pulled.round;
      renderBoundary(theta);

      // Fresh per-round, per-worker batch — labeled against the coord's current task
      const batchSeed = ((myRound + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
      const batch = makeBatch(batchSeed, BATCH_SIZE, currentTask);

      const grad = computeGradient(theta, batch);

      const reported = await reportGradient(myRound, Array.from(grad));
      updateStats(reported);
      if (reported.advanced) {
        log(`R${myRound} done · loss ${reported.last_loss.toFixed(4)}`);
        await pullHistory();
      } else if (reported.rejected) {
        log(`R${myRound} stale — catching up to R${reported.round}`);
      } else {
        log(`R${myRound} waiting · pool ${reported.pool_size}/${reported.target}`);
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
  pullHistory();
  runForever();
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset coordinator state? All progress lost.")) return;
  await fetch("/api/reset", { method: "POST" });
  history = [];
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
    await fetch("/api/set_task", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task: newTask }),
    });
    history = [];
    currentTask = newTask;
    log(`task → ${newTask}`);
    await pullHistory();
  });
}

pullHistory();
setInterval(pullHistory, 5000);
