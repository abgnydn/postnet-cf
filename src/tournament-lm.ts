/**
 * Phase 5 — char-LM tournament.
 *
 * Same flip-and-accept protocol as the float and ternary tournaments,
 * but the model is a next-character predictor over a fixed text:
 *
 *   embed (V × E) + linear (E × V) + bias (V) = 27·16 + 16·27 + 27 = 891 params
 *
 * Loss: mean cross-entropy across every char position in TEXT.
 *
 * Same DO shape as Phase 2/3 Tournament: applied_since reconciliation,
 * R2 binary snapshot at /api/lm/snapshot.bin. Workers score proposals
 * via their own forward pass — same code path that BitNet would slot
 * into in Phase 6 (just swap forward() for fused-lora WebGPU inference).
 */
import { DurableObject } from "cloudflare:workers";

export interface Env {
  COORD: DurableObjectNamespace;
  TOURNAMENT: DurableObjectNamespace;
  TERNARY: DurableObjectNamespace;
  TOURNAMENT_LM: DurableObjectNamespace;
  ASSETS: Fetcher;
  SNAPSHOTS: R2Bucket;
}

const V = 27;             // a-z + space
const E = 16;             // embed dim
const P_EMBED = V * E;    // 432
const P_OUT = E * V;      // 432
const P_BIAS = V;         // 27
const P = P_EMBED + P_OUT + P_BIAS;   // 891
const TARGET_PROPOSALS = 2;
const FLIP_SIZE = 6;
const FLIP_SIGMA = 0.15;
const SNAPSHOT_EVERY = 50;
const SNAPSHOT_KEY_PREFIX = "lm/";
// Phase 6: sharded snapshots. Cloudflare Worker response cap is 100 MB —
// at BitNet 2B scale (~282 MB) the bootstrap must be split across multiple
// R2 keys and fetched in parallel. SHARD_SIZE = 1 KB here so even the
// 891-param demo splits into 4 shards and exercises the parallel path.
// In production with BitNet 2B you'd use SHARD_SIZE ≈ 64 MB → ~5 shards.
const SHARD_SIZE = 1024;
const FLOATS_PER_SHARD = SHARD_SIZE / 4;

// Toy training text — short enough to score quickly, long enough to have
// learnable bigram/trigram structure. ~340 chars.
const TEXT =
  "the bird sings every dawn the cat sleeps in the sun the dog runs to the park " +
  "the wind blows the leaves the rain falls in the night the moon shines bright " +
  "above the mountains rise high and the river flows fast through the woods and " +
  "over the stones to the wide open sea where the waves break and roll back " +
  "again and again forever and ever";

function charCode(c: string): number {
  if (c === ' ') return 0;
  const k = c.charCodeAt(0) - 97;
  return k >= 0 && k < 26 ? k + 1 : 0;
}

const CODES = new Uint8Array(TEXT.length);
for (let i = 0; i < TEXT.length; i++) CODES[i] = charCode(TEXT[i]);

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function forward(theta: Float32Array, charIdx: number, outLogits: Float32Array): void {
  // Embed: theta[0 ... V*E - 1] is V × E, row-major. embedding[c] = row c
  const eStart = charIdx * E;
  // Output: theta[V*E ... V*E + E*V - 1] is E × V, row-major.
  // logits[v] = bias[v] + sum_i embed[i] * theta[P_EMBED + i*V + v]
  for (let v = 0; v < V; v++) outLogits[v] = theta[P_EMBED + P_OUT + v];
  for (let i = 0; i < E; i++) {
    const ei = theta[eStart + i];
    for (let v = 0; v < V; v++) {
      outLogits[v] += ei * theta[P_EMBED + i * V + v];
    }
  }
}

function testLoss(theta: Float32Array): number {
  const logits = new Float32Array(V);
  let loss = 0;
  const eps = 1e-7;
  for (let i = 0; i < CODES.length - 1; i++) {
    forward(theta, CODES[i], logits);
    // softmax + cross-entropy
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[v] - mx);
    const target = CODES[i + 1];
    loss += -(logits[target] - mx - Math.log(sum + eps));
  }
  return loss / (CODES.length - 1);
}

interface Proposal {
  worker_id: string;
  indices: number[];
  values: number[];
  delta: number;
}
interface AppliedFlip {
  round: number;
  indices: number[];
  values: number[];
}

export class TournamentLM extends DurableObject<Env> {
  private round = 0;
  private theta: Float32Array;
  private bestProposal: Proposal | null = null;
  private proposalsReceived = 0;
  private joined = new Set<string>();
  private lastLoss = -1;
  private accepted = 0;
  private considered = 0;
  private history: { round: number; loss: number; accepted: boolean; delta: number; ts: number }[] = [];
  private appliedHistory: AppliedFlip[] = [];
  private snapshotRound = 0;
  private snapshotShards: string[] = [];   // R2 keys, one per shard
  private bornAt = Date.now();

  constructor(state: DurableObjectState, env: Env) {
    super(state, env);
    const rng = mulberry32(7);
    this.theta = new Float32Array(P);
    for (let i = 0; i < P; i++) this.theta[i] = (rng() - 0.5) * 0.3;
    this.lastLoss = testLoss(this.theta);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: this.bornAt });
    state.blockConcurrencyWhile(async () => { await this.publishSnapshot(); });
  }

  private async publishSnapshot(): Promise<void> {
    // Phase 6: shard θ into SHARD_SIZE chunks, one R2 key per shard.
    // First shard carries an 8-byte header [round, P]; subsequent shards
    // are raw float32 contiguous slices. Workers fetch all shards in parallel.
    const numShards = Math.ceil(P / FLOATS_PER_SHARD);
    const shardKeys: string[] = [];
    const puts: Promise<unknown>[] = [];
    for (let k = 0; k < numShards; k++) {
      const floatStart = k * FLOATS_PER_SHARD;
      const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, P);
      const floatCount = floatEnd - floatStart;
      const headerSize = k === 0 ? 8 : 0;
      const buf = new ArrayBuffer(headerSize + floatCount * 4);
      const view = new DataView(buf);
      if (k === 0) {
        view.setUint32(0, this.round, true);
        view.setUint32(4, P, true);
      }
      new Float32Array(buf, headerSize, floatCount).set(this.theta.subarray(floatStart, floatEnd));
      const key = `${SNAPSHOT_KEY_PREFIX}r${this.round}/shard${k}.bin`;
      shardKeys.push(key);
      puts.push(this.env.SNAPSHOTS.put(key, buf, {
        httpMetadata: { contentType: "application/octet-stream" },
        customMetadata: {
          round: String(this.round),
          p: String(P),
          shard: String(k),
          total_shards: String(numShards),
          kind: "lm",
        },
      }).catch(() => {}));
    }
    try {
      await Promise.all(puts);
      this.snapshotRound = this.round;
      this.snapshotShards = shardKeys;
    } catch {}
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/lm/tick" && request.method === "POST") return this.tick(request);
    if (url.pathname === "/api/lm/state") return Response.json(this.summary());
    if (url.pathname === "/api/lm/sample") {
      const seedChar = url.searchParams.get("seed") || "t";
      const n = parseInt(url.searchParams.get("n") || "80");
      return Response.json({ sample: this.sampleText(seedChar, n) });
    }
    if (url.pathname === "/api/lm/snapshot") {
      // Phase 6: return a shard manifest. Worker fetches all shards in parallel.
      const round = this.snapshotRound || this.round;
      const numShards = Math.ceil(P / FLOATS_PER_SHARD);
      const shards = [];
      for (let k = 0; k < numShards; k++) {
        const floatStart = k * FLOATS_PER_SHARD;
        const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, P);
        const floatCount = floatEnd - floatStart;
        const headerSize = k === 0 ? 8 : 0;
        shards.push({
          url: `/api/lm/snapshot.bin?round=${round}&shard=${k}`,
          shard: k,
          float_start: floatStart,
          float_count: floatCount,
          bytes: headerSize + floatCount * 4,
        });
      }
      return Response.json({
        round, P, V, E,
        num_shards: numShards,
        shard_size_floats: FLOATS_PER_SHARD,
        shards,
        snapshot_bytes_total: 8 + P * 4,
      });
    }
    if (url.pathname === "/api/lm/snapshot.bin") {
      const wantRound = parseInt(url.searchParams.get("round") || "-1");
      const wantShard = parseInt(url.searchParams.get("shard") || "0");
      // Try R2 first
      if (this.snapshotShards.length > 0
          && (wantRound < 0 || wantRound === this.snapshotRound)
          && wantShard >= 0 && wantShard < this.snapshotShards.length) {
        const obj = await this.env.SNAPSHOTS.get(this.snapshotShards[wantShard]);
        if (obj) {
          return new Response(obj.body, {
            headers: {
              "content-type": "application/octet-stream",
              "x-snapshot-round": String(this.snapshotRound),
              "x-snapshot-shard": String(wantShard),
              "x-snapshot-source": "r2",
            },
          });
        }
      }
      // In-memory fallback for the requested shard
      const numShards = Math.ceil(P / FLOATS_PER_SHARD);
      const shardIdx = Math.max(0, Math.min(wantShard, numShards - 1));
      const floatStart = shardIdx * FLOATS_PER_SHARD;
      const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, P);
      const floatCount = floatEnd - floatStart;
      const headerSize = shardIdx === 0 ? 8 : 0;
      const buf = new ArrayBuffer(headerSize + floatCount * 4);
      const view = new DataView(buf);
      if (shardIdx === 0) {
        view.setUint32(0, this.round, true);
        view.setUint32(4, P, true);
      }
      new Float32Array(buf, headerSize, floatCount).set(this.theta.subarray(floatStart, floatEnd));
      return new Response(buf, {
        headers: {
          "content-type": "application/octet-stream",
          "x-snapshot-round": String(this.round),
          "x-snapshot-shard": String(shardIdx),
          "x-snapshot-source": "memory",
        },
      });
    }
    if (url.pathname === "/api/lm/reset" && request.method === "POST") {
      await this.resetState();
      return Response.json({ ok: true });
    }
    return new Response("not found", { status: 404 });
  }

  private async resetState() {
    const rng = mulberry32(Math.floor(Math.random() * 0xFFFFFFFF));
    this.theta = new Float32Array(P);
    for (let i = 0; i < P; i++) this.theta[i] = (rng() - 0.5) * 0.3;
    this.round = 0;
    this.bestProposal = null;
    this.proposalsReceived = 0;
    this.joined.clear();
    this.accepted = 0;
    this.considered = 0;
    this.history = [];
    this.appliedHistory = [];
    this.snapshotRound = 0;
    this.snapshotShards = [];
    this.lastLoss = testLoss(this.theta);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: Date.now() });
    await this.publishSnapshot();
  }

  private sampleText(seedChar: string, n: number): string {
    let c = charCode(seedChar);
    const logits = new Float32Array(V);
    const out: number[] = [c];
    const rng = mulberry32(Date.now() & 0xFFFFFFFF);
    for (let k = 0; k < n; k++) {
      forward(this.theta, c, logits);
      // Greedy + small noise for variety
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

  private async tick(request: Request): Promise<Response> {
    const body = await request.json<{
      worker_id: string;
      round?: number;
      indices?: number[];
      values?: number[];
      delta?: number;
      since_round?: number;
    }>();
    if (!body.worker_id) return new Response("missing worker_id", { status: 400 });
    this.joined.add(body.worker_id);

    let accepted = false, rejected = false;
    if (Array.isArray(body.indices) && Array.isArray(body.values) && typeof body.delta === "number") {
      if (body.round === this.round && body.indices.length === body.values.length) {
        this.considered += 1;
        if (!this.bestProposal || body.delta < this.bestProposal.delta) {
          this.bestProposal = {
            worker_id: body.worker_id,
            indices: body.indices,
            values: body.values,
            delta: body.delta,
          };
        }
        this.proposalsReceived += 1;
        accepted = true;
      } else {
        rejected = true;
      }
    }

    let advanced = false;
    let appliedDelta = 0;
    let appliedFlip: AppliedFlip | null = null;
    if (this.proposalsReceived >= TARGET_PROPOSALS) {
      const apply = this.bestProposal && this.bestProposal.delta < 0;
      if (apply) {
        for (let i = 0; i < this.bestProposal!.indices.length; i++) {
          const idx = this.bestProposal!.indices[i];
          if (idx >= 0 && idx < P) this.theta[idx] = this.bestProposal!.values[i];
        }
        this.accepted += 1;
        appliedDelta = this.bestProposal!.delta;
        appliedFlip = {
          round: this.round,
          indices: this.bestProposal!.indices.slice(),
          values: this.bestProposal!.values.slice(),
        };
        this.appliedHistory.push(appliedFlip);
        if (this.appliedHistory.length > 1000) this.appliedHistory.shift();
        if (this.accepted > 0 && this.accepted % SNAPSHOT_EVERY === 0) {
          this.ctx.waitUntil(this.publishSnapshot());
        }
      }
      this.lastLoss = testLoss(this.theta);
      this.history.push({ round: this.round, loss: this.lastLoss, accepted: !!apply, delta: appliedDelta, ts: Date.now() });
      if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
      this.round += 1;
      this.bestProposal = null;
      this.proposalsReceived = 0;
      advanced = true;
    }

    let appliedSince: AppliedFlip[] | null = null;
    if (typeof body.since_round === "number") {
      appliedSince = this.appliedHistory.filter(f => f.round >= body.since_round!);
    }
    const oldestAppliedRound = this.appliedHistory.length > 0 ? this.appliedHistory[0].round : null;

    return Response.json({
      round: this.round,
      P, V, E,
      flip_size: FLIP_SIZE,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: this.joined.size,
      last_loss: this.lastLoss,
      last_applied: appliedFlip,
      applied_since: appliedSince,
      oldest_applied_round: oldestAppliedRound,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      accepted, rejected, advanced,
    });
  }

  private summary() {
    return {
      round: this.round,
      P, V, E,
      flip_size: FLIP_SIZE,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: Array.from(this.joined),
      last_loss: this.lastLoss,
      accepted: this.accepted,
      considered: this.considered,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      history: this.history.slice(-200),
      sample: this.sampleText("t", 60),
      text_preview: TEXT.slice(0, 80),
      uptime_ms: Date.now() - this.bornAt,
    };
  }
}
