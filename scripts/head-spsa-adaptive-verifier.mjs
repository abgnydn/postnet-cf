// Phase 39 — head-classifier SPSA + adaptive-η verifier.
// Wire format identical to head-spsa-verifier.mjs. The only difference is the
// server endpoint: /api/head-spsa-adaptive/* updates η after each apply
// (multiplicative trust region on real_Δ; see src/tournament-head-spsa-adaptive.ts).
// Workers pull `eta` from every /tick response and use it for both the trial
// step and the apply reconstruction — so η-history is implicitly synced.

const D = 384, H = 128, K = 4;
const P_FC1 = D * H, P_B1 = H, P_FC2 = H * K, P_B2 = K;
const P = P_FC1 + P_B1 + P_FC2 + P_B2;   // 49796
const FC1_OFF = 0;
const B1_OFF = FC1_OFF + P_FC1;
const FC2_OFF = B1_OFF + P_B1;
const B2_OFF = FC2_OFF + P_FC2;

const TRAIN_N = 75;
const EPSILON_DEFAULT = 0.005;
const ETA_DEFAULT = 0.001;
const TRIALS = parseInt(process.env.TRIALS || "4");
const COORD = process.env.COORD || "http://localhost:8787";
const N_WORKERS = 3;
const ROUNDS = parseInt(process.env.ROUNDS || "600");

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

// MUST be bit-identical to src/tournament-head-spsa.ts
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

async function loadDataset() {
  const r = await fetch(`${COORD}/data/agnews-mini.bin`);
  if (!r.ok) throw new Error(`dataset fetch ${r.status}`);
  const buf = await r.arrayBuffer();
  const view = new DataView(buf);
  const N = view.getUint32(0, true);
  const headerBytes = 16;
  return {
    N,
    features: new Float32Array(buf, headerBytes, N * D).slice(),
    labels: new Uint8Array(buf, headerBytes + N * D * 4, N).slice(),
  };
}

function shardForWorkerIdx(idx) {
  const span = Math.floor(TRAIN_N / N_WORKERS);
  const start = idx * span;
  const end = idx === N_WORKERS - 1 ? TRAIN_N : (idx + 1) * span;
  return { start, end, idx };
}

class SpsaWorker {
  constructor(id, shard, features, labels) {
    this.id = id;
    this.shard = shard;
    this.features = features;
    this.labels = labels;
    this.localTheta = null;
    this.localRound = -1;
    this.bytesUp = 0;
    this.bytesDown = 0;
    this.epsilon = EPSILON_DEFAULT;
    this.eta = ETA_DEFAULT;
    this.uScratch = new Float32Array(P);
    this.thetaPlus = new Float32Array(P);
    this.thetaMinus = new Float32Array(P);
    this.thetaStep = new Float32Array(P);
  }
  async fetchJson(url, init) {
    const body = init && init.body ? init.body : "";
    this.bytesUp += Buffer.byteLength(body);
    for (let a = 0; a < 3; a++) {
      try {
        const r = await fetch(url, init);
        const text = await r.text();
        this.bytesDown += Buffer.byteLength(text);
        return JSON.parse(text);
      } catch (e) {
        if (a === 2) throw e;
        await new Promise(r => setTimeout(r, 50 * (a + 1)));
      }
    }
  }
  async bootstrap() {
    const meta = await this.fetchJson(`${COORD}/api/head-spsa-adaptive/snapshot`);
    if (typeof meta.epsilon === "number") this.epsilon = meta.epsilon;
    if (typeof meta.eta === "number") this.eta = meta.eta;
    const buffers = await Promise.all(meta.shards.map(async s => {
      const r = await fetch(`${COORD}${s.url}`);
      const buf = await r.arrayBuffer();
      this.bytesDown += buf.byteLength;
      return { shard: s.shard, headerSize: s.shard === 0 ? 8 : 0, buf };
    }));
    buffers.sort((a, b) => a.shard - b.shard);
    const headView = new DataView(buffers[0].buf);
    this.localRound = headView.getUint32(0, true);
    const p = headView.getUint32(4, true);
    if (p !== P) throw new Error(`P mismatch: server=${p} client=${P}`);
    this.localTheta = new Float32Array(P);
    for (const b of buffers) {
      const start = b.shard * (meta.shard_size_floats || (b.buf.byteLength - b.headerSize) / 4);
      this.localTheta.set(new Float32Array(b.buf, b.headerSize), start);
    }
  }
  async tick(round, seed, scalar_g, delta) {
    const body = { worker_id: this.id, since_round: this.localRound };
    if (round != null) {
      body.round = round;
      body.seed = seed;
      body.scalar_g = scalar_g;
      body.delta = delta;
    }
    return await this.fetchJson(`${COORD}/api/head-spsa-adaptive/tick`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }
  applyDelta(applied) {
    reconstructPerturbation(applied.seed, this.uScratch);
    const k = this.eta * applied.scalar_g;
    for (let i = 0; i < P; i++) this.localTheta[i] -= k * this.uScratch[i];
  }
  async reconcile(resp) {
    if (typeof resp.epsilon === "number") this.epsilon = resp.epsilon;
    if (typeof resp.eta === "number") this.eta = resp.eta;
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
  spsaTrial(rng, lossBefore) {
    const seed = (rng() * 0xFFFFFFFF) >>> 0;
    reconstructPerturbation(seed, this.uScratch);
    for (let i = 0; i < P; i++) {
      this.thetaPlus[i] = this.localTheta[i] + this.epsilon * this.uScratch[i];
      this.thetaMinus[i] = this.localTheta[i] - this.epsilon * this.uScratch[i];
    }
    const lossPlus = classLoss(this.thetaPlus, this.features, this.labels, this.shard.start, this.shard.end);
    const lossMinus = classLoss(this.thetaMinus, this.features, this.labels, this.shard.start, this.shard.end);
    const scalar_g = (lossPlus - lossMinus) / (2 * this.epsilon);
    const k = this.eta * scalar_g;
    for (let i = 0; i < P; i++) this.thetaStep[i] = this.localTheta[i] - k * this.uScratch[i];
    const lossAt = classLoss(this.thetaStep, this.features, this.labels, this.shard.start, this.shard.end);
    return { seed, scalar_g, delta: lossAt - lossBefore };
  }
}

async function runWorker(worker, stopAt) {
  await worker.bootstrap();
  const isByzantine = worker.id.includes("byz");
  while (true) {
    const pulled = await worker.tick();
    if (pulled.round >= stopAt) return;
    await worker.reconcile(pulled);
    const seedBase = ((worker.localRound + 1) * 1000003) ^
                     (worker.id.charCodeAt(0) * 31 + worker.id.charCodeAt(2));
    const rng = mulberry32(seedBase);
    let best;
    if (isByzantine) {
      const fakeSeed = (rng() * 0xFFFFFFFF) >>> 0;
      best = { seed: fakeSeed, scalar_g: 1.0, delta: -10 };
    } else {
      const lossBefore = classLoss(worker.localTheta, worker.features, worker.labels, worker.shard.start, worker.shard.end);
      best = null;
      for (let t = 0; t < TRIALS; t++) {
        const trial = worker.spsaTrial(rng, lossBefore);
        if (!best || trial.delta < best.delta) best = trial;
      }
    }
    const reported = await worker.tick(worker.localRound, best.seed, best.scalar_g, best.delta);
    await worker.reconcile(reported);
    await new Promise(r => setTimeout(r, 2));
  }
}

(async () => {
  console.log(`Phase 38 head-SPSA federated tournament · ${N_WORKERS} workers · ${ROUNDS} rounds`);
  console.log(`P=${P} D=${D} H=${H} K=${K} · trials=${TRIALS} · ε=${EPSILON_DEFAULT} η=${ETA_DEFAULT}`);
  console.log(`Random-init loss ≈ log(K) = ${Math.log(K).toFixed(3)} · random acc = ${(1/K*100).toFixed(0)}%\n`);
  await fetch(`${COORD}/api/head-spsa-adaptive/reset`, { method: "POST" });
  const data = await loadDataset();
  console.log(`Dataset loaded: N=${data.N} (TRAIN=${TRAIN_N}, TEST=${data.N - TRAIN_N})\n`);

  const workerIds = ["alpha", "delta", "bravo"];
  if (process.env.BYZANTINE === "1") workerIds.push("byz");
  const workers = workerIds.map((suffix, i) => new SpsaWorker(
    `head-spsa-adaptive-${suffix}`, shardForWorkerIdx(i % N_WORKERS), data.features, data.labels));
  for (const w of workers) {
    console.log(`  ${w.id} → shard ${w.shard.idx} examples[${w.shard.start}..${w.shard.end})`);
  }
  console.log();
  const running = workers.map(w => runWorker(w, ROUNDS));
  let lastShown = -1;
  const sampler = (async () => {
    while (true) {
      const s = await (await fetch(`${COORD}/api/head-spsa-adaptive/state`)).json();
      if (s.round !== lastShown && (s.round % 50 === 0)) {
        const etaStr = typeof s.eta === "number" ? s.eta.toExponential(2) : "—";
        const growShr = (typeof s.eta_grow_events === "number") ? ` grow=${s.eta_grow_events} shr=${s.eta_shrink_events ?? "?"}` : "";
        console.log(`R${String(s.round).padStart(4)}  loss=${s.last_loss.toFixed(4)}  acc=${(s.last_acc*100).toFixed(1)}%  ar=${(s.accept_rate*100).toFixed(0)}%  η=${etaStr}${growShr}`);
        lastShown = s.round;
      }
      if (s.round >= ROUNDS) return s;
      await new Promise(r => setTimeout(r, 100));
    }
  })();
  await Promise.all(running);
  const final = await sampler;
  const totalDown = workers.reduce((a, w) => a + w.bytesDown, 0);
  console.log(`\nfinal R${final.round}  loss=${final.last_loss.toFixed(4)}  acc=${(final.last_acc*100).toFixed(1)}%  accept=${(final.accept_rate*100).toFixed(1)}%`);
  console.log(`bandwidth: ${(totalDown/1024).toFixed(1)} KB down`);
  if (final.worker_stats) {
    console.log("\nworker_stats:");
    for (const [wid, st] of Object.entries(final.worker_stats)) {
      const tag = st.fraud_rate > 0.5 ? " ← BYZANTINE" : "";
      console.log(`  ${wid.padEnd(18)} wins=${String(st.wins).padStart(4)} frauds=${String(st.frauds).padStart(4)} (${(st.fraud_rate*100).toFixed(1)}%)${tag}`);
    }
  }
})();
