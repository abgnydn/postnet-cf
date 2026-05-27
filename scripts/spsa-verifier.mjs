// Phase 36 — char-LM SPSA tournament verifier.
// 3 workers each propose K=4 SPSA trials per round on the 2 379-param model
// and report their best (seed, scalar_g, delta). Mirror of lm-verifier.mjs.
//
// Phase 0 acceptance gate (CLAUDE.md): SPSA reaches the same final loss as
// flip-and-accept in ≤ 2× rounds; byzantine attack mode is quarantined.

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

const EPSILON_DEFAULT = 0.01;
const ETA_DEFAULT = 0.005;
const TRIALS = parseInt(process.env.TRIALS || "4");
const COORD = process.env.COORD || "http://localhost:8787";
const N_WORKERS = 3;
const ROUNDS = parseInt(process.env.ROUNDS || "1500");

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

// MUST be bit-identical to src/tournament-spsa-lm.ts and spsa-lm-worker.js.
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

class SpsaWorker {
  constructor(id) {
    this.id = id;
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
    const meta = await this.fetchJson(`${COORD}/api/spsa-lm/snapshot`);
    if (!Array.isArray(meta.shards)) throw new Error("no shards");
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
  async tick(round, seed, scalar_g, delta) {
    const body = { worker_id: this.id, since_round: this.localRound };
    if (round != null) {
      body.round = round;
      body.seed = seed;
      body.scalar_g = scalar_g;
      body.delta = delta;
    }
    return await this.fetchJson(`${COORD}/api/spsa-lm/tick`, {
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
  spsaTrial(rng, lossBefore, shard) {
    const seed = (rng() * 0xFFFFFFFF) >>> 0;
    reconstructPerturbation(seed, this.uScratch);
    for (let i = 0; i < P; i++) {
      this.thetaPlus[i] = this.localTheta[i] + this.epsilon * this.uScratch[i];
      this.thetaMinus[i] = this.localTheta[i] - this.epsilon * this.uScratch[i];
    }
    const lossPlus = textLoss(this.thetaPlus, shard.start, shard.end);
    const lossMinus = textLoss(this.thetaMinus, shard.start, shard.end);
    const scalar_g = (lossPlus - lossMinus) / (2 * this.epsilon);
    const k = this.eta * scalar_g;
    for (let i = 0; i < P; i++) this.thetaStep[i] = this.localTheta[i] - k * this.uScratch[i];
    const lossAt = textLoss(this.thetaStep, shard.start, shard.end);
    return { seed, scalar_g, delta: lossAt - lossBefore };
  }
}

async function runWorker(worker, stopAt) {
  await worker.bootstrap();
  const shard = shardForWorker(worker.id);
  worker.shard = shard;
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
      const lossBefore = textLoss(worker.localTheta, shard.start, shard.end);
      best = null;
      for (let t = 0; t < TRIALS; t++) {
        const trial = worker.spsaTrial(rng, lossBefore, shard);
        if (!best || trial.delta < best.delta) best = trial;
      }
    }
    const reported = await worker.tick(worker.localRound, best.seed, best.scalar_g, best.delta);
    await worker.reconcile(reported);
    await new Promise(r => setTimeout(r, 2));
  }
}

(async () => {
  console.log(`Phase 36 char-LM SPSA tournament · ${N_WORKERS} workers · ${ROUNDS} rounds`);
  console.log(`P=${P} V=${V} E=${E} · trials=${TRIALS} · ε=${EPSILON_DEFAULT} η=${ETA_DEFAULT}`);
  console.log(`Random-init loss ≈ log(V) = ${Math.log(V).toFixed(3)}\n`);

  await fetch(`${COORD}/api/spsa-lm/reset`, { method: "POST" });
  const workerIds = ["alpha", "delta", "bravo"];
  if (process.env.BYZANTINE === "1") workerIds.push("byz");
  const workers = workerIds.map(suffix => new SpsaWorker(`spsa-lm-${suffix}`));
  for (const w of workers) {
    const sh = shardForWorker(w.id);
    console.log(`  ${w.id} → shard${sh.idx} [${sh.start}..${sh.end}]`);
  }
  const running = workers.map(w => runWorker(w, ROUNDS));
  let lastShown = -1;
  const sampler = (async () => {
    while (true) {
      const s = await (await fetch(`${COORD}/api/spsa-lm/state`)).json();
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
  console.log(`bandwidth: ${(totalDown/1024).toFixed(1)} KB down`);
  console.log(`final sample: ${JSON.stringify(final.sample.slice(0, 100))}`);
  if (final.worker_stats) {
    console.log("\nworker_stats:");
    for (const [wid, st] of Object.entries(final.worker_stats)) {
      const tag = st.fraud_rate > 0.5 ? " ← BYZANTINE" : "";
      console.log(`  ${wid.padEnd(16)} wins=${String(st.wins).padStart(4)} frauds=${String(st.frauds).padStart(4)} (${(st.fraud_rate*100).toFixed(1)}%)${tag}`);
    }
  }
})();
