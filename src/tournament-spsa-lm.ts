/**
 * Phase 36 — char-LM tournament, SPSA-flavored.
 *
 * Forks tournament-lm.ts. Same model (V=27 / E=16 / HID=32 / CTX=2, P=2379),
 * same text, same byzantine defense, same sharded snapshot path. The ONLY
 * thing that changes is the proposal shape and the apply math:
 *
 *   flip-and-accept (TournamentLM)        SPSA tournament (this DO)
 *   ─────────────────────────────        ────────────────────────────
 *   {indices[6], values[6], delta}        {seed, scalar_g, delta}
 *   ~52 B wire per proposal               20 B wire per proposal
 *   θ[idx_k] = val_k  for k=0..5          θ ← θ − η · scalar_g · u(seed)
 *   "tweak 6 random params"               "step ε in a coherent Gaussian
 *                                          direction across ALL P params"
 *
 * The "real_Δ vs claimed_Δ" fraud check is byte-identical — that's the whole
 * point of this fusion. See docs/PROTOCOL.md when promoted to v0.6.
 *
 * Why this exists: random flip-and-accept stops finding improvements past
 * P ~ 10K. SPSA's directional derivative scales to LLM-class P (cf. MeZO,
 * DeComFL, FedKSeed). This DO is the smallest possible test of the swap
 * on the existing char-LM, side-by-side with the flip-and-accept baseline.
 */
import { DurableObject } from "cloudflare:workers";

export interface Env {
  COORD: DurableObjectNamespace;
  TOURNAMENT: DurableObjectNamespace;
  TERNARY: DurableObjectNamespace;
  TOURNAMENT_LM: DurableObjectNamespace;
  TOURNAMENT_SPSA_LM: DurableObjectNamespace;
  ASSETS: Fetcher;
  SNAPSHOTS: R2Bucket;
}

// Architecture: identical to TournamentLM. Side-by-side comparison requires
// the model be the same.
const V = 27;
const E = 16;
const HID = 32;
const CTX = 2;
const P_EMBED = V * E;                  // 432
const P_FC1 = CTX * E * HID;            // 1024
const P_B1 = HID;                       // 32
const P_FC2 = HID * V;                  // 864
const P_B2 = V;                         // 27
const P = P_EMBED + P_FC1 + P_B1 + P_FC2 + P_B2;   // 2379
const FC1_OFF = P_EMBED;
const B1_OFF = FC1_OFF + P_FC1;
const FC2_OFF = B1_OFF + P_B1;
const B2_OFF = FC2_OFF + P_FC2;

// SPSA hyperparameters. Exposed in /tick response so workers stay in sync
// even if these get retuned at runtime.
const EPSILON = 0.01;      // perturbation magnitude (probe radius)
const ETA = 0.005;         // step size
const TARGET_PROPOSALS = 2;
const SNAPSHOT_EVERY = 50;
const SNAPSHOT_KEY_PREFIX = "spsa-lm/";
const SHARD_SIZE = 1024;
const FLOATS_PER_SHARD = SHARD_SIZE / 4;

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

// CRITICAL: This function MUST be bit-identical between worker and server.
// Workers ship a Float32-typed perturbation reconstruction in JS that uses
// the same mulberry32+Box-Muller pair. Any drift here breaks the protocol.
// Output: Float32Array[P], each component i.i.d. N(0,1).
function reconstructPerturbation(seed: number, out: Float32Array): void {
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

function forward(theta: Float32Array, prevPrev: number, prev: number, outLogits: Float32Array): void {
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
    const tgt = CODES[i];
    loss += -(logits[tgt] - mx - Math.log(sum + eps));
  }
  return loss / (CODES.length - CTX);
}

interface SpsaProposal {
  worker_id: string;
  seed: number;
  scalar_g: number;
  delta: number;
}
interface SpsaAppliedFlip {
  round: number;
  seed: number;
  scalar_g: number;
}

export class TournamentSpsaLM extends DurableObject<Env> {
  private round = 0;
  private theta: Float32Array;
  private bestProposal: SpsaProposal | null = null;
  private proposalsReceived = 0;
  private joined = new Set<string>();
  private lastLoss = -1;
  private accepted = 0;
  private considered = 0;
  private history: { round: number; loss: number; accepted: boolean; delta: number; ts: number }[] = [];
  private appliedHistory: SpsaAppliedFlip[] = [];
  private snapshotRound = 0;
  private snapshotShards: string[] = [];
  private workerStats = new Map<string, {
    wins: number;
    frauds: number;
    lastWinRound: number;
    recent: number[];
    recent_long: number[];
  }>();
  private subscribers = new Set<WebSocket>();
  private bornAt = Date.now();
  // Scratch buffer for perturbation reconstruction. Lives on the DO so we
  // don't allocate P*4 bytes every apply.
  private uScratch: Float32Array;

  constructor(state: DurableObjectState, env: Env) {
    super(state, env);
    const rng = mulberry32(7);   // same seed as TournamentLM for fair compare
    this.theta = new Float32Array(P);
    for (let i = 0; i < P; i++) this.theta[i] = (rng() - 0.5) * 0.3;
    this.uScratch = new Float32Array(P);
    this.lastLoss = testLoss(this.theta);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: this.bornAt });
    state.blockConcurrencyWhile(async () => { await this.publishSnapshot(); });
  }

  private async publishSnapshot(): Promise<void> {
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
          kind: "spsa-lm",
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
    if (url.pathname === "/api/spsa-lm/tick" && request.method === "POST") return this.tick(request);
    if (url.pathname === "/api/spsa-lm/state") return Response.json(this.summary());
    if (url.pathname === "/api/spsa-lm/sample") {
      const seedChar = url.searchParams.get("seed") || "t";
      const n = parseInt(url.searchParams.get("n") || "80");
      return Response.json({ sample: this.sampleText(seedChar, n) });
    }
    if (url.pathname === "/api/spsa-lm/snapshot") {
      const round = this.snapshotRound || this.round;
      const numShards = Math.ceil(P / FLOATS_PER_SHARD);
      const shards = [];
      for (let k = 0; k < numShards; k++) {
        const floatStart = k * FLOATS_PER_SHARD;
        const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, P);
        const floatCount = floatEnd - floatStart;
        const headerSize = k === 0 ? 8 : 0;
        shards.push({
          url: `/api/spsa-lm/snapshot.bin?round=${round}&shard=${k}`,
          shard: k,
          float_start: floatStart,
          float_count: floatCount,
          bytes: headerSize + floatCount * 4,
        });
      }
      return Response.json({
        round, P, V, E, HID, CTX,
        epsilon: EPSILON,
        eta: ETA,
        num_shards: numShards,
        shard_size_floats: FLOATS_PER_SHARD,
        shards,
        snapshot_bytes_total: 8 + P * 4,
      });
    }
    if (url.pathname === "/api/spsa-lm/snapshot.bin") {
      const wantRound = parseInt(url.searchParams.get("round") || "-1");
      const wantShard = parseInt(url.searchParams.get("shard") || "0");
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
    if (url.pathname === "/api/spsa-lm/reset" && request.method === "POST") {
      await this.resetState();
      return Response.json({ ok: true });
    }
    if (url.pathname === "/api/spsa-lm/ws") {
      const upgrade = request.headers.get("upgrade");
      if (upgrade !== "websocket") return new Response("expected websocket", { status: 426 });
      const pair = new WebSocketPair();
      const [client, server] = Object.values(pair);
      server.accept();
      this.subscribers.add(server);
      server.send(JSON.stringify({
        type: "hello",
        round: this.round,
        last_loss: this.lastLoss,
        recent: this.appliedHistory.slice(-50),
        epsilon: EPSILON,
        eta: ETA,
      }));
      server.addEventListener("close", () => { this.subscribers.delete(server); });
      server.addEventListener("error", () => { this.subscribers.delete(server); });
      return new Response(null, { status: 101, webSocket: client });
    }
    return new Response("not found", { status: 404 });
  }

  private broadcast(msg: object) {
    const text = JSON.stringify(msg);
    for (const ws of this.subscribers) {
      try { ws.send(text); } catch { this.subscribers.delete(ws); }
    }
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
    let prevPrev = 0;
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

  // The one math change from TournamentLM: apply an SPSA step instead of
  // an index-flip. Reconstruct u from the seed (bit-identical to worker),
  // then  θ ← θ − η · scalar_g · u.
  private applySpsaStep(seed: number, scalar_g: number): void {
    reconstructPerturbation(seed, this.uScratch);
    const k = ETA * scalar_g;
    for (let i = 0; i < P; i++) this.theta[i] -= k * this.uScratch[i];
  }

  private async tick(request: Request): Promise<Response> {
    const body = await request.json<{
      worker_id: string;
      round?: number;
      seed?: number;
      scalar_g?: number;
      delta?: number;
      since_round?: number;
    }>();
    if (!body.worker_id) return new Response("missing worker_id", { status: 400 });
    this.joined.add(body.worker_id);

    let accepted = false, rejected = false, quarantined = false;
    const hasProposal =
      typeof body.seed === "number" &&
      typeof body.scalar_g === "number" &&
      typeof body.delta === "number";
    if (hasProposal) {
      if (body.round === this.round) {
        // Same tiered defense as TournamentLM, byte-for-byte.
        const stats = this.workerStats.get(body.worker_id);
        const cumRate = stats && stats.wins >= 10 ? stats.frauds / stats.wins : 0;
        const recent = stats?.recent ?? [];
        const recentLong = stats?.recent_long ?? [];
        const winRate = recent.length >= 10
          ? recent.reduce((a, b) => a + b, 0) / recent.length
          : 0;
        const longRate = recentLong.length >= 30
          ? recentLong.reduce((a, b) => a + b, 0) / recentLong.length
          : 0;
        const quarantineHit =
          cumRate > 0.4 ||
          winRate > 0.4 ||
          longRate > 0.25;
        if (quarantineHit) {
          quarantined = true;
        } else {
          this.considered += 1;
          if (!this.bestProposal || body.delta! < this.bestProposal.delta) {
            this.bestProposal = {
              worker_id: body.worker_id,
              seed: body.seed!,
              scalar_g: body.scalar_g!,
              delta: body.delta!,
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
    let appliedFlip: SpsaAppliedFlip | null = null;
    if (this.proposalsReceived >= TARGET_PROPOSALS) {
      const apply = this.bestProposal && this.bestProposal.delta < 0;
      const lossBefore = this.lastLoss;
      if (apply) {
        this.applySpsaStep(this.bestProposal!.seed, this.bestProposal!.scalar_g);
        this.accepted += 1;
        appliedDelta = this.bestProposal!.delta;
        appliedFlip = {
          round: this.round,
          seed: this.bestProposal!.seed,
          scalar_g: this.bestProposal!.scalar_g,
        };
        this.appliedHistory.push(appliedFlip);
        if (this.appliedHistory.length > 1000) this.appliedHistory.shift();
        if (this.accepted > 0 && this.accepted % SNAPSHOT_EVERY === 0) {
          this.ctx.waitUntil(this.publishSnapshot());
        }
      }
      this.lastLoss = testLoss(this.theta);
      // Byzantine fraud detection — IDENTICAL to TournamentLM. The post-apply
      // real_Δ check works because we always recompute lastLoss on the full
      // text after applying, regardless of what kind of update was applied.
      if (apply && this.bestProposal) {
        const realGlobalDelta = this.lastLoss - lossBefore;
        const winnerId = this.bestProposal.worker_id;
        const stats = this.workerStats.get(winnerId) ?? { wins: 0, frauds: 0, lastWinRound: -1, recent: [], recent_long: [] };
        stats.wins += 1;
        stats.lastWinRound = this.round;
        const isFraud = realGlobalDelta > 1e-4 && this.bestProposal.delta < -1e-4;
        if (isFraud) stats.frauds += 1;
        stats.recent.push(isFraud ? 1 : 0);
        if (stats.recent.length > 20) stats.recent.shift();
        stats.recent_long.push(isFraud ? 1 : 0);
        if (stats.recent_long.length > 100) stats.recent_long.shift();
        this.workerStats.set(winnerId, stats);
      }
      this.history.push({ round: this.round, loss: this.lastLoss, accepted: !!apply, delta: appliedDelta, ts: Date.now() });
      if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
      this.round += 1;
      this.bestProposal = null;
      this.proposalsReceived = 0;
      advanced = true;
      this.broadcast({
        type: "advance",
        round: this.round,
        last_loss: this.lastLoss,
        applied: appliedFlip,
      });
    }

    let appliedSince: SpsaAppliedFlip[] | null = null;
    if (typeof body.since_round === "number") {
      appliedSince = this.appliedHistory.filter(f => f.round >= body.since_round!);
    }
    const oldestAppliedRound = this.appliedHistory.length > 0 ? this.appliedHistory[0].round : null;

    return Response.json({
      round: this.round,
      P, V, E, HID, CTX,
      epsilon: EPSILON,
      eta: ETA,
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
      epsilon: EPSILON,
      eta: ETA,
      worker_stats: workerReport,
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
