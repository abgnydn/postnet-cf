// Phase 4 — ternary-weight tournament verifier.
// Workers maintain a local Int8Array sign[] + scalar scale, fetch the
// packed binary snapshot, propose ternary flips (each position → new
// value in {-1, 0, +1}), report the best.

const H = 32, P = 4 * H + 1;
const BATCH = 128;
const TRIALS = 8;
const FLIP_SIZE = 6;
const COORD = process.env.COORD || "http://localhost:8787";
const N_WORKERS = 3;
const ROUNDS_PER_TASK = 800;   // ternary needs more rounds — coarser landscape

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

function forward(sign, scale, x, y) {
  let z = sign[4 * H] * scale;
  const w2Off = 3 * H, b1Off = 2 * H;
  for (let i = 0; i < H; i++) {
    let h = (sign[i*2]*x + sign[i*2+1]*y) * scale + sign[b1Off+i] * scale;
    if (h < 0) h = 0;
    z += sign[w2Off + i] * scale * h;
  }
  return 1 / (1 + Math.exp(-z));
}

function batchLoss(sign, scale, batch) {
  let loss = 0;
  const N = batch.xs.length, eps = 1e-7;
  for (let i = 0; i < N; i++) {
    const p = forward(sign, scale, batch.xs[i], batch.ys[i]);
    const y = batch.labels[i];
    loss += -(y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps));
  }
  return loss / N;
}

function unpackTernary(packed, p) {
  const out = new Int8Array(p);
  for (let i = 0; i < p; i++) {
    const byteIdx = (i * 2) >> 3;
    const bitOff = (i * 2) & 7;
    const code = (packed[byteIdx] >> bitOff) & 0b11;
    out[i] = code === 0 ? 0 : code === 1 ? 1 : -1;
  }
  return out;
}

function proposeFlip(sign, rng) {
  // Pick FLIP_SIZE unique positions, propose new ternary values ≠ current
  const indices = new Int32Array(FLIP_SIZE);
  const values = new Int8Array(FLIP_SIZE);
  const seen = new Set();
  for (let k = 0; k < FLIP_SIZE; k++) {
    let idx;
    do { idx = Math.floor(rng() * P); } while (seen.has(idx));
    seen.add(idx);
    indices[k] = idx;
    // Choose new value uniformly from {-1, 0, +1} \ {current}
    const cur = sign[idx];
    const choices = [-1, 0, 1].filter(v => v !== cur);
    values[k] = choices[Math.floor(rng() * choices.length)];
  }
  return { indices, values };
}

class Worker {
  constructor(id) {
    this.id = id;
    this.sign = null;
    this.scale = 0.5;
    this.localRound = -1;
    this.bytesUp = 0;
    this.bytesDown = 0;
    this.bootstraps = 0;
  }
  async fetchJson(url, init) {
    const body = init && init.body ? init.body : "";
    this.bytesUp += Buffer.byteLength(body);
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const r = await fetch(url, init);
        const text = await r.text();
        this.bytesDown += Buffer.byteLength(text);
        return JSON.parse(text);
      } catch (e) {
        if (attempt === 2) throw e;
        await new Promise(r => setTimeout(r, 50 * (attempt + 1)));
      }
    }
  }
  async bootstrap() {
    const meta = await this.fetchJson(`${COORD}/api/ternary/snapshot`);
    const r = await fetch(`${COORD}${meta.snapshot_url}`);
    const buf = await r.arrayBuffer();
    this.bytesDown += buf.byteLength;
    const view = new DataView(buf);
    const round = view.getUint32(0, true);
    const p = view.getUint32(4, true);
    if (p !== P) throw new Error(`P mismatch: server=${p} client=${P}`);
    this.scale = view.getFloat32(8, true);
    const packed = new Uint8Array(buf, 12);
    this.sign = unpackTernary(packed, p);
    this.localRound = round;
    this.bootstraps += 1;
    return meta;
  }
  async tick(round, indices, values, delta) {
    const body = { worker_id: this.id, since_round: this.localRound };
    if (round != null) {
      body.round = round;
      body.indices = Array.from(indices);
      body.values = Array.from(values);
      body.delta = delta;
    }
    return await this.fetchJson(`${COORD}/api/ternary/tick`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }
  applyDelta(applied) {
    for (let k = 0; k < applied.indices.length; k++) {
      this.sign[applied.indices[k]] = applied.values[k];
    }
  }
  async reconcile(resp) {
    if (resp.round === this.localRound) return;
    if (typeof resp.oldest_applied_round === "number"
        && resp.oldest_applied_round > this.localRound + 1) {
      await this.bootstrap();
      return;
    }
    if (Array.isArray(resp.applied_since)) {
      for (const flip of resp.applied_since) {
        if (flip.round >= this.localRound) this.applyDelta(flip);
      }
    }
    this.localRound = resp.round;
  }
}

async function setTask(task) {
  const r = await fetch(`${COORD}/api/ternary/set_task`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ task }),
  });
  return r.json();
}

async function runWorker(worker, task, stopAt) {
  await worker.bootstrap();
  while (true) {
    const pulled = await worker.tick();
    if (pulled.round >= stopAt) return;
    if (pulled.task !== task) { await new Promise(r => setTimeout(r, 20)); continue; }
    await worker.reconcile(pulled);

    const seed = ((worker.localRound + 1) * 1000003) ^ (worker.id.charCodeAt(0) * 31 + worker.id.charCodeAt(2));
    const rng = mulberry32(seed);
    const batch = makeBatch(seed ^ 0xA17BEEF, BATCH, task);
    const lossBefore = batchLoss(worker.sign, worker.scale, batch);
    const trial = new Int8Array(P);
    let best = null;
    for (let t = 0; t < TRIALS; t++) {
      const { indices, values } = proposeFlip(worker.sign, rng);
      for (let i = 0; i < P; i++) trial[i] = worker.sign[i];
      for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
      const lossAfter = batchLoss(trial, worker.scale, batch);
      const delta = lossAfter - lossBefore;
      if (!best || delta < best.delta) best = { indices, values, delta };
    }
    const reported = await worker.tick(worker.localRound, best.indices, best.values, best.delta);
    await worker.reconcile(reported);
    await new Promise(r => setTimeout(r, 2));
  }
}

async function runTask(task) {
  console.log(`\n=== ${task.toUpperCase()} (ternary) ===`);
  const r = await setTask(task);
  console.log(`reset → task=${r.task}`);
  const workers = Array.from({ length: N_WORKERS }, (_, i) => new Worker(`tn-${task}-${i}`));
  const running = workers.map(w => runWorker(w, task, ROUNDS_PER_TASK));
  let lastShown = -1;
  const sampler = (async () => {
    while (true) {
      const s = await (await fetch(`${COORD}/api/ternary/state`)).json();
      if (s.round !== lastShown && s.round % 200 === 0) {
        console.log(`R${s.round}\tloss=${s.last_loss.toFixed(4)}\taccept=${(s.accept_rate*100).toFixed(1)}%\tpeers=${s.joined.length}`);
        lastShown = s.round;
      }
      if (s.round >= ROUNDS_PER_TASK) return s;
      await new Promise(r => setTimeout(r, 100));
    }
  })();
  await Promise.all(running);
  const final = await sampler;
  const totalUp = workers.reduce((a, w) => a + w.bytesUp, 0);
  const totalDown = workers.reduce((a, w) => a + w.bytesDown, 0);
  console.log(`final R${final.round}\tloss=${final.last_loss.toFixed(4)}\taccept=${(final.accept_rate*100).toFixed(1)}%`);
  console.log(`bandwidth (3 workers): up=${(totalUp/1024).toFixed(1)} KB · down=${(totalDown/1024).toFixed(1)} KB`);
  return { loss: final.last_loss, accept: final.accept_rate };
}

(async () => {
  console.log(`Phase 4 ternary verifier · ${N_WORKERS} workers · ${ROUNDS_PER_TASK} rounds × 3 tasks`);
  console.log(`weights ∈ {-1, 0, +1} × scale · K=${TRIALS} trials · flip_size=${FLIP_SIZE} · batch=${BATCH}`);
  const results = {};
  for (const task of ["circle", "xor", "wave"]) {
    results[task] = await runTask(task);
  }
  console.log("\n=== SUMMARY ===");
  for (const [task, r] of Object.entries(results)) {
    const verdict = r.loss < 0.25 ? "✓ converged" : r.loss < 0.4 ? "△ partial" : "✗ stuck";
    console.log(`${task.padEnd(8)} loss ${r.loss.toFixed(4)}  accept ${(r.accept*100).toFixed(1).padStart(5)}%  ${verdict}`);
  }
})();
