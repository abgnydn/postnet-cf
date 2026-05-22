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

// Phase 8: context-2 MLP. Previous 2 chars → embed (shared) → concat → MLP → next char.
// Architecture: embed(V × E) + fc1(2E × H) + b1(H) + fc2(H × V) + b2(V).
const V = 27;             // a-z + space
const E = 16;             // embed dim per position
const HID = 32;           // hidden width
const CTX = 2;            // context length (previous CTX chars predict the next)
const P_EMBED = V * E;                  // 432
const P_FC1 = CTX * E * HID;            // 1024
const P_B1 = HID;                       // 32
const P_FC2 = HID * V;                  // 864
const P_B2 = V;                         // 27
const P = P_EMBED + P_FC1 + P_B1 + P_FC2 + P_B2;   // 2379
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

// Phase 8: context-2 forward.
// theta layout:
//   [0 .. P_EMBED)                      : embed (V × E)
//   [P_EMBED .. P_EMBED + P_FC1)        : fc1 (2E × H), row-major (2E rows, H cols)
//   [P_EMBED + P_FC1 .. + P_B1)         : b1 (H)
//   [... .. + P_FC2)                    : fc2 (H × V)
//   [... .. + P_B2)                     : b2 (V)
const FC1_OFF = P_EMBED;
const B1_OFF = FC1_OFF + P_FC1;
const FC2_OFF = B1_OFF + P_B1;
const B2_OFF = FC2_OFF + P_FC2;

function forward(theta: Float32Array, prevPrev: number, prev: number, outLogits: Float32Array): void {
  // Concat two embeddings to form a 2E-vector x
  const x = new Float32Array(CTX * E);
  for (let i = 0; i < E; i++) {
    x[i] = theta[prevPrev * E + i];
    x[E + i] = theta[prev * E + i];
  }
  // fc1: h_j = relu(b1_j + sum_i x_i * theta[FC1_OFF + i*H + j])
  const h = new Float32Array(HID);
  for (let j = 0; j < HID; j++) h[j] = theta[B1_OFF + j];
  for (let i = 0; i < CTX * E; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = FC1_OFF + i * HID;
    for (let j = 0; j < HID; j++) h[j] += xi * theta[row + j];
  }
  for (let j = 0; j < HID; j++) if (h[j] < 0) h[j] = 0;
  // fc2: logits[v] = b2_v + sum_j h_j * theta[FC2_OFF + j*V + v]
  for (let v = 0; v < V; v++) outLogits[v] = theta[B2_OFF + v];
  for (let j = 0; j < HID; j++) {
    const hj = h[j];
    if (hj === 0) continue;
    const row = FC2_OFF + j * V;
    for (let v = 0; v < V; v++) outLogits[v] += hj * theta[row + v];
  }
}

function testLoss(theta: Float32Array): number {
  const logits = new Float32Array(V);
  let loss = 0;
  const eps = 1e-7;
  for (let i = CTX; i < CODES.length; i++) {
    forward(theta, CODES[i - 2], CODES[i - 1], logits);
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[v] - mx);
    const target = CODES[i];
    loss += -(logits[target] - mx - Math.log(sum + eps));
  }
  return loss / (CODES.length - CTX);
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
  // Phase 9: byzantine fraud tracking. Honest workers report delta ≈ scaled
  // version of the global delta (they only see a shard, but signs should agree).
  // A worker whose winning proposals actually *increase* full-text loss is
  // suspect — they're claiming improvements that aren't there.
  private workerStats = new Map<string, { wins: number; frauds: number; lastWinRound: number }>();
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
        round, P, V, E, HID, CTX,
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
    this.workerStats.clear();
    this.lastLoss = testLoss(this.theta);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: Date.now() });
    await this.publishSnapshot();
  }

  private sampleText(seedChar: string, n: number): string {
    let prevPrev = 0;   // start with space as context
    let prev = charCode(seedChar);
    const logits = new Float32Array(V);
    const out: number[] = [prev];
    const rng = mulberry32(Date.now() & 0xFFFFFFFF);
    for (let k = 0; k < n; k++) {
      forward(this.theta, prevPrev, prev, logits);
      let mx = -Infinity, arg = 0;
      for (let v = 0; v < V; v++) {
        const score = logits[v] + (rng() - 0.5) * 0.3;
        if (score > mx) { mx = score; arg = v; }
      }
      prevPrev = prev;
      prev = arg;
      out.push(arg);
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

    let accepted = false, rejected = false, quarantined = false;
    if (Array.isArray(body.indices) && Array.isArray(body.values) && typeof body.delta === "number") {
      if (body.round === this.round && body.indices.length === body.values.length) {
        // Phase 9: quarantine high-fraud workers. After 10 wins, if > 40% of
        // them increased the global loss, skip their proposals entirely.
        const stats = this.workerStats.get(body.worker_id);
        const fraudRate = stats && stats.wins >= 10 ? stats.frauds / stats.wins : 0;
        if (fraudRate > 0.4) {
          quarantined = true;
        } else {
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
        }
      } else {
        rejected = true;
      }
    }

    let advanced = false;
    let appliedDelta = 0;
    let appliedFlip: AppliedFlip | null = null;
    if (this.proposalsReceived >= TARGET_PROPOSALS) {
      const apply = this.bestProposal && this.bestProposal.delta < 0;
      const lossBefore = this.lastLoss;
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
      // Phase 9: Byzantine fraud detection. A winning worker who claimed
      // a negative delta but whose flip actually raised the global loss is
      // suspect. Honest workers may have noisy estimates from sharded data
      // but their sign should align with the global change.
      if (apply && this.bestProposal) {
        const realGlobalDelta = this.lastLoss - lossBefore;
        const winnerId = this.bestProposal.worker_id;
        const stats = this.workerStats.get(winnerId) ?? { wins: 0, frauds: 0, lastWinRound: -1 };
        stats.wins += 1;
        stats.lastWinRound = this.round;
        if (realGlobalDelta > 1e-4 && this.bestProposal.delta < -1e-4) stats.frauds += 1;
        this.workerStats.set(winnerId, stats);
      }
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
      P, V, E, HID, CTX,
      flip_size: FLIP_SIZE,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: this.joined.size,
      last_loss: this.lastLoss,
      last_applied: appliedFlip,
      applied_since: appliedSince,
      oldest_applied_round: oldestAppliedRound,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      accepted, rejected, quarantined, advanced,
    });
  }

  private summary() {
    const workerReport: Record<string, { wins: number; frauds: number; fraud_rate: number; lastWinRound: number }> = {};
    for (const [wid, stats] of this.workerStats.entries()) {
      workerReport[wid] = {
        wins: stats.wins,
        frauds: stats.frauds,
        fraud_rate: stats.wins > 0 ? stats.frauds / stats.wins : 0,
        lastWinRound: stats.lastWinRound,
      };
    }
    return {
      round: this.round,
      P, V, E, HID, CTX,
      worker_stats: workerReport,
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
