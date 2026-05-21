// Headless multi-task verifier.
// Acts as N parallel browser workers against the local coord;
// runs wave/circle/xor sequentially and reports loss trajectory per task.

const H = 32, P = 4 * H + 1, BATCH = 64;
const COORD = process.env.COORD || "http://localhost:8787";
const N_WORKERS = 3;
const ROUNDS_PER_TASK = 120;

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

function computeGradient(theta, batch) {
  const grad = new Float32Array(P);
  const N = batch.xs.length;
  const w2Off = 3 * H, b1Off = 2 * H, b2Off = 4 * H;
  const pre = new Float32Array(H), h = new Float32Array(H);
  for (let n = 0; n < N; n++) {
    const x = batch.xs[n], y = batch.ys[n], label = batch.labels[n];
    let z = theta[b2Off];
    for (let i = 0; i < H; i++) {
      const v = theta[i*2]*x + theta[i*2+1]*y + theta[b1Off+i];
      pre[i] = v;
      const hv = v > 0 ? v : 0;
      h[i] = hv;
      z += theta[w2Off + i] * hv;
    }
    const p = 1 / (1 + Math.exp(-z));
    const dz = p - label;
    grad[b2Off] += dz;
    for (let i = 0; i < H; i++) {
      grad[w2Off + i] += dz * h[i];
      if (pre[i] > 0) {
        const dpre = dz * theta[w2Off + i];
        grad[i*2] += dpre * x;
        grad[i*2+1] += dpre * y;
        grad[b1Off + i] += dpre;
      }
    }
  }
  const invN = 1 / N;
  for (let i = 0; i < P; i++) grad[i] *= invN;
  return grad;
}

async function tick(workerId, round, gradient) {
  const body = { worker_id: workerId };
  if (gradient != null) { body.round = round; body.gradient = Array.from(gradient); }
  const r = await fetch(`${COORD}/api/tick`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function setTask(task) {
  const r = await fetch(`${COORD}/api/set_task`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ task }),
  });
  return r.json();
}

async function runWorker(workerId, task, stopAt) {
  let lastReportedRound = -1;
  while (true) {
    const pulled = await tick(workerId);
    if (pulled.round >= stopAt) return;
    if (pulled.task !== task) { await new Promise(r => setTimeout(r, 20)); continue; }
    const theta = new Float32Array(pulled.theta);
    const seed = ((pulled.round + 1) * 1000003) ^ (workerId.charCodeAt(0) * 31 + workerId.charCodeAt(2));
    const batch = makeBatch(seed, BATCH, task);
    const grad = computeGradient(theta, batch);
    await tick(workerId, pulled.round, grad);
    lastReportedRound = pulled.round;
    await new Promise(r => setTimeout(r, 5));
  }
}

async function runTask(task) {
  console.log(`\n=== ${task.toUpperCase()} ===`);
  const r = await setTask(task);
  console.log(`reset → task=${r.task}`);
  const workers = Array.from({ length: N_WORKERS }, (_, i) =>
    runWorker(`w-${task}-${i}`, task, ROUNDS_PER_TASK));
  // Sampler: every 20 rounds, log current loss
  let lastShown = -1;
  const sampler = (async () => {
    while (true) {
      const s = await (await fetch(`${COORD}/api/state`)).json();
      if (s.round !== lastShown && s.round % 20 === 0) {
        console.log(`R${s.round}\tloss=${s.last_loss.toFixed(4)}\tpeers=${s.joined.length}`);
        lastShown = s.round;
      }
      if (s.round >= ROUNDS_PER_TASK) return s;
      await new Promise(r => setTimeout(r, 100));
    }
  })();
  await Promise.all(workers);
  const final = await sampler;
  console.log(`final R${final.round}\tloss=${final.last_loss.toFixed(4)}`);
  return final.last_loss;
}

(async () => {
  console.log(`Multi-task headless verifier · ${N_WORKERS} workers · ${ROUNDS_PER_TASK} rounds each`);
  const results = {};
  for (const task of ["circle", "xor", "wave"]) {
    results[task] = await runTask(task);
  }
  console.log("\n=== SUMMARY ===");
  for (const [task, loss] of Object.entries(results)) {
    const verdict = loss < 0.2 ? "✓ converged" : loss < 0.35 ? "△ partial" : "✗ stuck";
    console.log(`${task.padEnd(8)} final loss ${loss.toFixed(4)}  ${verdict}`);
  }
})();
