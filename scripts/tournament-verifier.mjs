// Headless Phase-1 verifier — tournament / flip-and-accept protocol.
// Each "worker" locally proposes K flips, scores them on its private batch,
// reports its best. The coord picks the best across all workers per round.

const H = 32, P = 4 * H + 1;
const BATCH = 128;
const TRIALS = 8;
const FLIP_SIZE = 4;
const FLIP_SIGMA = 0.15;
const COORD = process.env.COORD || "http://localhost:8787";
const N_WORKERS = 3;
const ROUNDS_PER_TASK = 400;

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
  if (task === "xor") return ((x > 0) !== (y > 0)) ? 1 : 0;
  return Math.sin(2 * x) > y ? 1 : 0;
}
function makeBatch(seed, n, task) {
  const rng = mulberry32(seed);
  const xs = new Float32Array(n), ys = new Float32Array(n), labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = (rng() - 0.5) * 4;
    ys[i] = (rng() - 0.5) * 4;
    labels[i] = trueLabel(task, xs[i], ys[i]);
  }
  return { xs, ys, labels };
}
function forward(theta, x, y) {
  let z = theta[4 * H];
  const w2Off = 3 * H, b1Off = 2 * H;
  for (let i = 0; i < H; i++) {
    let h = theta[i*2]*x + theta[i*2+1]*y + theta[b1Off+i];
    if (h < 0) h = 0;
    z += theta[w2Off + i] * h;
  }
  return 1 / (1 + Math.exp(-z));
}
function batchLoss(theta, batch) {
  let loss = 0;
  const N = batch.xs.length, eps = 1e-7;
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

async function tick(workerId, round, indices, values, delta) {
  const body = { worker_id: workerId };
  if (round != null) { body.round = round; body.indices = Array.from(indices); body.values = Array.from(values); body.delta = delta; }
  const r = await fetch(`${COORD}/api/tournament/tick`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}
async function setTask(task) {
  const r = await fetch(`${COORD}/api/tournament/set_task`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ task }),
  });
  return r.json();
}

async function runWorker(workerId, task, stopAt) {
  while (true) {
    const pulled = await tick(workerId);
    if (pulled.round >= stopAt) return;
    if (pulled.task !== task) { await new Promise(r => setTimeout(r, 20)); continue; }
    const theta = new Float32Array(pulled.theta);
    const seed = ((pulled.round + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
    const rng = mulberry32(seed);
    const batch = makeBatch(seed ^ 0xA17BEEF, BATCH, task);
    const lossBefore = batchLoss(theta, batch);
    const trial = new Float32Array(P);
    let best = null;
    for (let t = 0; t < TRIALS; t++) {
      const { indices, values } = proposeFlip(theta, rng);
      for (let i = 0; i < P; i++) trial[i] = theta[i];
      for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
      const lossAfter = batchLoss(trial, batch);
      const delta = lossAfter - lossBefore;
      if (!best || delta < best.delta) best = { indices, values, delta };
    }
    await tick(workerId, pulled.round, best.indices, best.values, best.delta);
    await new Promise(r => setTimeout(r, 2));
  }
}

async function runTask(task) {
  console.log(`\n=== ${task.toUpperCase()} (tournament) ===`);
  const r = await setTask(task);
  console.log(`reset → task=${r.task}`);
  const workers = Array.from({ length: N_WORKERS }, (_, i) =>
    runWorker(`tv-${task}-${i}`, task, ROUNDS_PER_TASK));
  let lastShown = -1;
  const sampler = (async () => {
    while (true) {
      const s = await (await fetch(`${COORD}/api/tournament/state`)).json();
      if (s.round !== lastShown && s.round % 50 === 0) {
        console.log(`R${s.round}\tloss=${s.last_loss.toFixed(4)}\taccept=${(s.accept_rate*100).toFixed(1)}%\tpeers=${s.joined.length}`);
        lastShown = s.round;
      }
      if (s.round >= ROUNDS_PER_TASK) return s;
      await new Promise(r => setTimeout(r, 100));
    }
  })();
  await Promise.all(workers);
  const final = await sampler;
  console.log(`final R${final.round}\tloss=${final.last_loss.toFixed(4)}\taccept=${(final.accept_rate*100).toFixed(1)}%`);
  return { loss: final.last_loss, accept: final.accept_rate };
}

(async () => {
  console.log(`Phase 1 tournament verifier · ${N_WORKERS} workers · ${ROUNDS_PER_TASK} rounds × 3 tasks`);
  console.log(`K=${TRIALS} trials per report · flip_size=${FLIP_SIZE} · σ=${FLIP_SIGMA} · batch=${BATCH}`);
  const results = {};
  for (const task of ["circle", "xor", "wave"]) {
    results[task] = await runTask(task);
  }
  console.log("\n=== SUMMARY ===");
  for (const [task, r] of Object.entries(results)) {
    const verdict = r.loss < 0.2 ? "✓ converged" : r.loss < 0.35 ? "△ partial" : "✗ stuck";
    console.log(`${task.padEnd(8)} loss ${r.loss.toFixed(4)}  accept ${(r.accept*100).toFixed(1).padStart(5)}%  ${verdict}`);
  }
})();
