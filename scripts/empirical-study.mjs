// Phase 12: Multi-seed empirical study of the char-LM tournament protocol.
// Runs N independent training sessions per variant (resetting between each)
// and reports mean ± stddev final loss + a one-line t-test-style comparison.
//
// Variants exercised here:
//   - vanilla        : 3 honest workers, full text (no shards)
//   - sharded        : 3 workers, 3 disjoint text shards (Phase 7)
//   - byzantine-off  : 3 honest + 1 attacker, defense disabled at server  (NB: server-side; not user-toggleable here, only logs the run)
//   - byzantine-on   : 3 honest + 1 attacker, defense active (Phase 9)
//
// Each session runs ROUNDS_PER_RUN coord rounds. Coord resets between runs.

const COORD = process.env.COORD || "http://localhost:8787";
const N_SEEDS = parseInt(process.env.SEEDS || "3");
const ROUNDS_PER_RUN = parseInt(process.env.ROUNDS || "1500");

const V = 27, E = 16, HID = 32, CTX = 2;
const P_EMBED = V * E, P_FC1 = CTX * E * HID, P_B1 = HID, P_FC2 = HID * V, P_B2 = V;
const P = P_EMBED + P_FC1 + P_B1 + P_FC2 + P_B2;
const FC1_OFF = P_EMBED;
const B1_OFF = FC1_OFF + P_FC1;
const FC2_OFF = B1_OFF + P_B1;
const B2_OFF = FC2_OFF + P_FC2;
const TRIALS = 8, FLIP_SIZE = 6, FLIP_SIGMA = 0.15;

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
function forward(theta, pp, p, logits) {
  const x = new Float32Array(CTX * E);
  for (let i = 0; i < E; i++) { x[i] = theta[pp*E+i]; x[E+i] = theta[p*E+i]; }
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
function textLoss(theta, s, e) {
  const logits = new Float32Array(V);
  let loss = 0;
  const start = Math.max(s ?? 0, CTX);
  const end = e ?? CODES.length;
  for (let i = start; i < end; i++) {
    forward(theta, CODES[i - 2], CODES[i - 1], logits);
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[v] - mx);
    loss += -(logits[CODES[i]] - mx - Math.log(sum + 1e-7));
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

const NUM_SHARDS = 3;
function shardForWorker(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  const s = h % NUM_SHARDS;
  const span = Math.floor(CODES.length / NUM_SHARDS);
  return { start: s * span, end: s === NUM_SHARDS - 1 ? CODES.length : (s + 1) * span };
}

class Worker {
  constructor(id, variant, seedSuffix) {
    this.id = `${id}-s${seedSuffix}`;
    this.variant = variant;
    this.isByzantine = id.includes("byz");
    // Phase 14: smart byzantine — act honest for first SMART_HONEST_WINS wins
    // then attack. Tests sliding-window vs cumulative fraud detection.
    this.isSmartByz = id.includes("smart");
    this.smartWinsObserved = 0;
    this.useShards = variant === "sharded" || variant.startsWith("byzantine");
    this.localTheta = null;
    this.localRound = -1;
  }
  async fetchJson(url, init) {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const r = await fetch(url, init);
        return await r.json();
      } catch (e) {
        if (attempt === 2) throw e;
        await new Promise(r => setTimeout(r, 50 * (attempt + 1)));
      }
    }
  }
  async bootstrap() {
    const meta = await this.fetchJson(`${COORD}/api/lm/snapshot`);
    const buffers = await Promise.all(meta.shards.map(async s => {
      const r = await fetch(`${COORD}${s.url}`);
      return { shard: s.shard, headerSize: s.shard === 0 ? 8 : 0, buf: await r.arrayBuffer() };
    }));
    buffers.sort((a, b) => a.shard - b.shard);
    const head = new DataView(buffers[0].buf);
    this.localRound = head.getUint32(0, true);
    this.localTheta = new Float32Array(P);
    for (const b of buffers) {
      const start = b.shard * meta.shard_size_floats;
      const floats = new Float32Array(b.buf, b.headerSize);
      this.localTheta.set(floats, start);
    }
  }
  async tick(round, indices, values, delta) {
    const body = { worker_id: this.id, since_round: this.localRound };
    if (round != null) { body.round = round; body.indices = Array.from(indices); body.values = Array.from(values); body.delta = delta; }
    const r = await fetch(`${COORD}/api/lm/tick`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    return await r.json();
  }
  reconcile(resp) {
    if (resp.round === this.localRound) return;
    if (typeof resp.oldest_applied_round === "number" && resp.oldest_applied_round > this.localRound + 1) {
      return "bootstrap";
    }
    if (Array.isArray(resp.applied_since)) {
      for (const flip of resp.applied_since) {
        if (flip.round >= this.localRound) {
          for (let k = 0; k < flip.indices.length; k++) this.localTheta[flip.indices[k]] = flip.values[k];
        }
      }
    }
    this.localRound = resp.round;
  }
}

async function runWorker(worker, stopAt) {
  await worker.bootstrap();
  const trial = new Float32Array(P);
  const shard = worker.useShards ? shardForWorker(worker.id) : { start: 0, end: CODES.length };
  while (true) {
    const pulled = await worker.tick();
    if (pulled.round >= stopAt) return;
    const r = worker.reconcile(pulled);
    if (r === "bootstrap") { await worker.bootstrap(); continue; }
    const rng = mulberry32(((worker.localRound + 1) * 1000003) ^ (worker.id.charCodeAt(0) * 31 + worker.id.charCodeAt(2)));
    let best;
    if (worker.isSmartByz) {
      // First 9 wins: act honest (passes cumulative detection's >=10 threshold).
      // After that: attack. Sliding window over last 20 should catch this.
      const SMART_HONEST_WINS = 9;
      if (worker.smartWinsObserved < SMART_HONEST_WINS) {
        // Honest behavior
        const lossBefore = textLoss(worker.localTheta, shard.start, shard.end);
        best = null;
        for (let t = 0; t < TRIALS; t++) {
          const { indices, values } = proposeFlip(worker.localTheta, rng);
          for (let i = 0; i < P; i++) trial[i] = worker.localTheta[i];
          for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
          const delta = textLoss(trial, shard.start, shard.end) - lossBefore;
          if (!best || delta < best.delta) best = { indices, values, delta };
        }
      } else {
        const { indices, values } = proposeFlip(worker.localTheta, rng);
        best = { indices, values, delta: -10 };
      }
    } else if (worker.isByzantine) {
      const { indices, values } = proposeFlip(worker.localTheta, rng);
      best = { indices, values, delta: -10 };
    } else {
      const lossBefore = textLoss(worker.localTheta, shard.start, shard.end);
      best = null;
      for (let t = 0; t < TRIALS; t++) {
        const { indices, values } = proposeFlip(worker.localTheta, rng);
        for (let i = 0; i < P; i++) trial[i] = worker.localTheta[i];
        for (let k = 0; k < indices.length; k++) trial[indices[k]] = values[k];
        const delta = textLoss(trial, shard.start, shard.end) - lossBefore;
        if (!best || delta < best.delta) best = { indices, values, delta };
      }
    }
    const myIndices = best.indices;
    const submittedRound = worker.localRound;
    const reported = await worker.tick(worker.localRound, best.indices, best.values, best.delta);
    worker.reconcile(reported);
    // Phase 14: smart-byz win detection — check if the winning flip's indices match ours
    if (worker.isSmartByz && Array.isArray(reported.applied_since)) {
      for (const flip of reported.applied_since) {
        if (flip.round === submittedRound && flip.indices.length === myIndices.length
            && flip.indices.every((x, i) => x === myIndices[i])) {
          worker.smartWinsObserved += 1;
          break;
        }
      }
    }
    await new Promise(r => setTimeout(r, 1));
  }
}

async function runOneSession(variant, seedSuffix, attackerCount = 0, attackerKind = "dumb") {
  await fetch(`${COORD}/api/lm/reset`, { method: "POST" });
  await new Promise(r => setTimeout(r, 50));
  const honest = ["alpha", "delta", "bravo"];
  const attackerPrefix = attackerKind === "smart" ? "smart" : "byz";
  const attackers = ["0", "1", "2"].slice(0, attackerCount).map(n => `${attackerPrefix}${n}`);
  let workerIds;
  switch (variant) {
    case "vanilla":   workerIds = honest; break;
    case "sharded":   workerIds = honest; break;
    case "byzantine": workerIds = [...honest, ...attackers]; break;
    default: throw new Error("unknown variant: " + variant);
  }
  const workers = workerIds.map(s => new Worker(`lm-${s}`, variant, seedSuffix));
  const running = workers.map(w => runWorker(w, ROUNDS_PER_RUN));
  await Promise.all(running);
  const final = await (await fetch(`${COORD}/api/lm/state`)).json();
  const fraudByWorker = final.worker_stats || {};
  return { loss: final.last_loss, workerStats: fraudByWorker };
}

function meanStd(arr) {
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  const v = arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
  return { mean: m, std: Math.sqrt(v), n: arr.length };
}

(async () => {
  const mode = process.env.MODE || "variants";
  if (mode === "smart") {
    // Phase 14: compare dumb vs smart attacker at fixed count = 1
    const kinds = ["dumb", "smart"];
    console.log(`Smart-attacker comparison · ${N_SEEDS} seeds × ${ROUNDS_PER_RUN} rounds per cell\n`);
    console.log(`(3 honest + 1 attacker; smart acts honest for 9 wins then attacks)\n`);
    const results = {};
    for (const kind of kinds) {
      results[kind] = [];
      process.stdout.write(`${kind.padEnd(8)}`);
      for (let s = 0; s < N_SEEDS; s++) {
        const r = await runOneSession("byzantine", s, 1, kind);
        results[kind].push(r.loss);
        process.stdout.write(`  ${r.loss.toFixed(4)}`);
      }
      const ms = meanStd(results[kind]);
      console.log(`   →  ${ms.mean.toFixed(4)} ± ${ms.std.toFixed(4)}`);
    }
    console.log("\n=== SMART-ATTACKER SUMMARY ===");
    const base = meanStd(results.dumb);
    for (const kind of kinds) {
      const ms = meanStd(results[kind]);
      const delta = ms.mean - base.mean;
      const sign = delta >= 0 ? "+" : "";
      console.log(`${kind.padEnd(8)}  ${ms.n}  ${ms.mean.toFixed(4)}  ${ms.std.toFixed(4)}  ${sign}${delta.toFixed(4)}`);
    }
  } else if (mode === "attackers") {
    // Phase 13: sweep attacker count from 0 to 3 (honest count stays at 3)
    const counts = [0, 1, 2, 3];
    const results = {};
    console.log(`Attacker-count sweep · ${N_SEEDS} seeds × ${ROUNDS_PER_RUN} rounds per cell\n`);
    console.log(`(3 honest workers + N attackers per run)\n`);
    for (const k of counts) {
      results[k] = [];
      const label = `${k} atk`;
      process.stdout.write(`${label.padEnd(12)}`);
      for (let s = 0; s < N_SEEDS; s++) {
        const r = await runOneSession("byzantine", s, k);
        results[k].push(r.loss);
        process.stdout.write(`  ${r.loss.toFixed(4)}`);
      }
      const ms = meanStd(results[k]);
      const honest_share = 3 / (3 + k);
      console.log(`   →  ${ms.mean.toFixed(4)} ± ${ms.std.toFixed(4)}    honest_share=${(honest_share*100).toFixed(0)}%`);
    }
    console.log("\n=== ATTACKER-COUNT SUMMARY ===");
    console.log("attackers   n   mean    std     vs 0-atk");
    const baseMean = meanStd(results[0]).mean;
    for (const k of counts) {
      const ms = meanStd(results[k]);
      const delta = ms.mean - baseMean;
      const sign = delta >= 0 ? "+" : "";
      console.log(`${String(k).padStart(2)}          ${ms.n}  ${ms.mean.toFixed(4)}  ${ms.std.toFixed(4)}  ${sign}${delta.toFixed(4)}`);
    }
  } else {
    console.log(`Variant comparison · ${N_SEEDS} seeds × ${ROUNDS_PER_RUN} rounds per variant\n`);
    const variants = ["vanilla", "sharded", "byzantine"];
    const results = {};
    for (const v of variants) {
      results[v] = [];
      process.stdout.write(`${v.padEnd(12)}`);
      for (let s = 0; s < N_SEEDS; s++) {
        const r = await runOneSession(v, s, v === "byzantine" ? 1 : 0);
        results[v].push(r.loss);
        process.stdout.write(`  ${r.loss.toFixed(4)}`);
      }
      const ms = meanStd(results[v]);
      console.log(`   →  ${ms.mean.toFixed(4)} ± ${ms.std.toFixed(4)}`);
    }
    console.log("\n=== SUMMARY ===");
    console.log("variant       n   mean    std     vs vanilla");
    const vanillaMean = meanStd(results.vanilla).mean;
    for (const v of variants) {
      const ms = meanStd(results[v]);
      const delta = ms.mean - vanillaMean;
      const sign = delta >= 0 ? "+" : "";
      console.log(`${v.padEnd(12)}  ${ms.n}  ${ms.mean.toFixed(4)}  ${ms.std.toFixed(4)}  ${sign}${delta.toFixed(4)}`);
    }
  }
})();
