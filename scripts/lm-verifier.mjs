// Phase 5 — char-LM tournament verifier.
// 3 workers each proposes K=8 random-Gaussian flips per round on
// the 891-param next-char model and reports its best.

// Phase 8: context-2 MLP, must match server src/tournament-lm.ts
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
const TRIALS = 8;
const FLIP_SIZE = 6;
const FLIP_SIGMA = 0.15;
const COORD = process.env.COORD || "http://localhost:8787";
const N_WORKERS = 3;
const ROUNDS = parseInt(process.env.ROUNDS || "1500");

// Same TEXT as server for parity scoring on workers
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

class Worker {
  constructor(id) {
    this.id = id;
    this.localTheta = null;
    this.localRound = -1;
    this.bytesUp = 0;
    this.bytesDown = 0;
  }
  async fetchJson(url, init) {
    const body = init && init.body ? init.body : "";
    this.bytesUp += Buffer.byteLength(body);
    const r = await fetch(url, init);
    const text = await r.text();
    this.bytesDown += Buffer.byteLength(text);
    return JSON.parse(text);
  }
  async bootstrap() {
    // Phase 6: parallel shard fetch
    const meta = await this.fetchJson(`${COORD}/api/lm/snapshot`);
    if (!Array.isArray(meta.shards)) throw new Error("no shards");
    const buffers = await Promise.all(meta.shards.map(async s => {
      const r = await fetch(`${COORD}${s.url}`);
      const buf = await r.arrayBuffer();
      this.bytesDown += buf.byteLength;
      return { shard: s.shard, headerSize: s.shard === 0 ? 8 : 0, buf };
    }));
    buffers.sort((a, b) => a.shard - b.shard);
    const headView = new DataView(buffers[0].buf);
    const round = headView.getUint32(0, true);
    const p = headView.getUint32(4, true);
    if (p !== P) throw new Error(`P mismatch: server=${p} client=${P}`);
    this.localTheta = new Float32Array(P);
    for (const b of buffers) {
      const start = b.shard * (meta.shard_size_floats || (b.buf.byteLength - b.headerSize) / 4);
      const floats = new Float32Array(b.buf, b.headerSize);
      this.localTheta.set(floats, start);
    }
    this.localRound = round;
  }
  async tick(round, indices, values, delta) {
    const body = { worker_id: this.id, since_round: this.localRound };
    if (round != null) {
      body.round = round;
      body.indices = Array.from(indices);
      body.values = Array.from(values);
      body.delta = delta;
    }
    return await this.fetchJson(`${COORD}/api/lm/tick`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }
  applyDelta(applied) {
    for (let k = 0; k < applied.indices.length; k++) {
      this.localTheta[applied.indices[k]] = applied.values[k];
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

async function runWorker(worker, stopAt) {
  await worker.bootstrap();
  const trial = new Float32Array(P);
  const shard = shardForWorker(worker.id);
  worker.shard = shard;
  while (true) {
    const pulled = await worker.tick();
    if (pulled.round >= stopAt) return;
    await worker.reconcile(pulled);

    const seed = ((worker.localRound + 1) * 1000003) ^
                 (worker.id.charCodeAt(0) * 31 + worker.id.charCodeAt(2));
    const rng = mulberry32(seed);
    // Phase 7: score only on this worker's private shard
    const lossBefore = textLoss(worker.localTheta, shard.start, shard.end);
    let best = null;
    for (let t = 0; t < TRIALS; t++) {
      const { indices, values } = proposeFlip(worker.localTheta, rng);
      for (let i = 0; i < P; i++) trial[i] = worker.localTheta[i];
      for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
      const lossAfter = textLoss(trial, shard.start, shard.end);
      const delta = lossAfter - lossBefore;
      if (!best || delta < best.delta) best = { indices, values, delta };
    }
    const reported = await worker.tick(worker.localRound, best.indices, best.values, best.delta);
    await worker.reconcile(reported);
    await new Promise(r => setTimeout(r, 2));
  }
}

(async () => {
  console.log(`Phase 5 char-LM tournament · ${N_WORKERS} workers · ${ROUNDS} rounds`);
  console.log(`P=${P} V=${V} E=${E} · K=${TRIALS} trials · flip_size=${FLIP_SIZE} σ=${FLIP_SIGMA}`);
  console.log(`Random-init loss ≈ log(V) = ${Math.log(V).toFixed(3)}\n`);

  await fetch(`${COORD}/api/lm/reset`, { method: "POST" });
  // Phase 7: hand-pick worker ids so we hit each shard exactly once
  // alpha → shard 0, delta → shard 1, bravo → shard 2 (hand-picked for full coverage)
  const workers = ["alpha", "delta", "bravo"].map(suffix => new Worker(`lm-${suffix}`));
  for (const w of workers) {
    const sh = shardForWorker(w.id);
    console.log(`  ${w.id} → shard${sh.idx} [${sh.start}..${sh.end}]`);
  }
  const running = workers.map(w => runWorker(w, ROUNDS));
  let lastShown = -1;
  const sampler = (async () => {
    while (true) {
      const s = await (await fetch(`${COORD}/api/lm/state`)).json();
      if (s.round !== lastShown && s.round % 200 === 0) {
        const sample = s.sample ? s.sample.slice(0, 60).replace(/\n/g, '·') : '';
        console.log(`R${s.round}\tloss=${s.last_loss.toFixed(4)}\tac=${(s.accept_rate*100).toFixed(0)}%\tsample=${JSON.stringify(sample)}`);
        lastShown = s.round;
      }
      if (s.round >= ROUNDS) return s;
      await new Promise(r => setTimeout(r, 100));
    }
  })();
  await Promise.all(running);
  const final = await sampler;
  const totalDown = workers.reduce((a, w) => a + w.bytesDown, 0);
  console.log(`\nfinal R${final.round} loss=${final.last_loss.toFixed(4)} accept=${(final.accept_rate*100).toFixed(1)}%`);
  console.log(`bandwidth (3 workers): ${(totalDown/1024).toFixed(1)} KB down`);
  console.log(`final sample: ${JSON.stringify(final.sample.slice(0, 100))}`);
})();
