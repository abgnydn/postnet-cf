/**
 * Phase 39b — head-classifier tournament, SPSA + Adam-on-scalar.
 *
 * MEAZO-faithful sibling of Phase 39's symmetric AIMD. Server tracks Adam
 * moments OVER THE WINNING SCALAR g; on apply:
 *
 *   t  += 1
 *   m   = β₁·m + (1-β₁)·g
 *   v   = β₂·v + (1-β₂)·g²
 *   m̂   = m / (1 - β₁^t)
 *   v̂   = v / (1 - β₂^t)
 *   step_eff = lr · m̂ / (√v̂ + ε)
 *   θ  ← θ − step_eff · u(seed)
 *
 * Workers receive (m, v, t) in every /tick response and project their
 * PER-TRIAL step_eff_t using each candidate g_t, so worker and server
 * apply byte-identical updates for the winning proposal — byzantine
 * real_Δ check semantics unchanged.
 *
 * Wire format per proposal still 20 bytes; ~28 extra bytes per /tick
 * response for the Adam state.
 */
import { DurableObject } from "cloudflare:workers";
import {
  P, D, H, K,
  classLoss, classAccuracy, parseDataset, initTheta,
  CLASS_NAMES,
} from "./head-model";

export interface Env {
  COORD: DurableObjectNamespace;
  TOURNAMENT: DurableObjectNamespace;
  TERNARY: DurableObjectNamespace;
  TOURNAMENT_LM: DurableObjectNamespace;
  TOURNAMENT_SPSA_LM: DurableObjectNamespace;
  TOURNAMENT_LM_BIG: DurableObjectNamespace;
  TOURNAMENT_SPSA_LM_BIG: DurableObjectNamespace;
  TOURNAMENT_HEAD_FLIP: DurableObjectNamespace;
  TOURNAMENT_HEAD_SPSA: DurableObjectNamespace;
  TOURNAMENT_HEAD_SPSA_ADAPTIVE: DurableObjectNamespace;
  TOURNAMENT_HEAD_SPSA_ADAM: DurableObjectNamespace;
  ASSETS: Fetcher;
  SNAPSHOTS: R2Bucket;
}

const TRAIN_N = 75;
const EPSILON = 0.005;       // perturbation magnitude (same as Phase 38/39)
// Adam-on-scalar hyperparameters. lr=0.001 matches the Phase 38 fixed-η
// baseline so step_eff stays in a comparable magnitude family. β₁/β₂/ε
// are the textbook defaults.
const LR = 0.001;
const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS_ADAM = 1e-8;
const TARGET_PROPOSALS = 2;
const SNAPSHOT_EVERY = 50;
const SNAPSHOT_KEY_PREFIX = "head-spsa-adam/";
const SHARD_SIZE = 16 * 1024;
const FLOATS_PER_SHARD = SHARD_SIZE / 4;
const DATASET_URL = "http://internal/data/agnews-mini.bin";

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

// CRITICAL: must be bit-identical to scripts/head-spsa-verifier.mjs and
// public/head-spsa-worker.js. Server applies the worker's update by
// reconstructing u from this exact procedure.
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

export class TournamentHeadSpsaAdam extends DurableObject<Env> {
  private round = 0;
  private theta!: Float32Array;
  private features!: Float32Array;
  private labels!: Uint8Array;
  private N = 0;
  private testStart = 0;
  private testEnd = 0;
  private bestProposal: SpsaProposal | null = null;
  private proposalsReceived = 0;
  private joined = new Set<string>();
  private lastLoss = -1;
  private lastAcc = -1;
  private accepted = 0;
  private considered = 0;
  private history: { round: number; loss: number; acc: number; accepted: boolean; delta: number; ts: number }[] = [];
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
  private uScratch!: Float32Array;
  // Adaptive step size — the one thing that differs from TournamentHeadSpsa.
  // Workers read this from /tick + /snapshot every round and use it for both
  // Adam-on-scalar state: a single (m, v, t) tracking moments of the WINNING
  // scalar g across rounds. Workers receive (m, v, t) on every /tick + /snapshot
  // and project per-trial step_eff_t for their candidate g_t, so worker and
  // server compute byte-identical updates for the winning proposal.
  private adamM = 0;
  private adamV = 0;
  private adamT = 0;

  constructor(state: DurableObjectState, env: Env) {
    super(state, env);
    state.blockConcurrencyWhile(async () => {
      const r = await env.ASSETS.fetch(DATASET_URL);
      if (!r.ok) throw new Error(`dataset fetch ${r.status}`);
      const buf = await r.arrayBuffer();
      const parsed = parseDataset(buf);
      this.features = parsed.features;
      this.labels = parsed.labels;
      this.N = parsed.N;
      this.testStart = Math.min(TRAIN_N, this.N);
      this.testEnd = this.N;
      this.theta = initTheta(7);
      this.uScratch = new Float32Array(P);
      this.lastLoss = classLoss(this.theta, this.features, this.labels, this.testStart, this.testEnd);
      this.lastAcc = classAccuracy(this.theta, this.features, this.labels, this.testStart, this.testEnd);
      this.history.push({ round: -1, loss: this.lastLoss, acc: this.lastAcc, accepted: false, delta: 0, ts: this.bornAt });
      await this.publishSnapshot();
    });
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
          kind: "head-spsa-adam",
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
    if (url.pathname === "/api/head-spsa-adam/tick" && request.method === "POST") return this.tick(request);
    if (url.pathname === "/api/head-spsa-adam/state") return Response.json(this.summary());
    if (url.pathname === "/api/head-spsa-adam/snapshot") {
      const round = this.snapshotRound || this.round;
      const numShards = Math.ceil(P / FLOATS_PER_SHARD);
      const shards = [];
      for (let k = 0; k < numShards; k++) {
        const floatStart = k * FLOATS_PER_SHARD;
        const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, P);
        const floatCount = floatEnd - floatStart;
        const headerSize = k === 0 ? 8 : 0;
        shards.push({
          url: `/api/head-spsa-adam/snapshot.bin?round=${round}&shard=${k}`,
          shard: k,
          float_start: floatStart,
          float_count: floatCount,
          bytes: headerSize + floatCount * 4,
        });
      }
      return Response.json({
        round, P, D, H, K,
        epsilon: EPSILON,
        lr: LR, beta1: BETA1, beta2: BETA2, eps_adam: EPS_ADAM,
        adam_m: this.adamM, adam_v: this.adamV, adam_t: this.adamT,
        train_n: TRAIN_N,
        test_n: this.testEnd - this.testStart,
        num_shards: numShards,
        shard_size_floats: FLOATS_PER_SHARD,
        shards,
        snapshot_bytes_total: 8 + P * 4,
      });
    }
    if (url.pathname === "/api/head-spsa-adam/snapshot.bin") {
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
    if (url.pathname === "/api/head-spsa-adam/reset" && request.method === "POST") {
      await this.resetState();
      return Response.json({ ok: true });
    }
    if (url.pathname === "/api/head-spsa-adam/ws") {
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
        last_acc: this.lastAcc,
        recent: this.appliedHistory.slice(-50),
        epsilon: EPSILON,
        lr: LR, beta1: BETA1, beta2: BETA2, eps_adam: EPS_ADAM,
        adam_m: this.adamM, adam_v: this.adamV, adam_t: this.adamT,
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
    // Fixed seed so flip and SPSA start from the *same* θ for fair head-to-head
    // (matching changes in TournamentHeadFlip.resetState + TournamentHeadSpsa).
    this.theta = initTheta(7);
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
    this.adamM = 0;
    this.adamV = 0;
    this.adamT = 0;
    this.lastLoss = classLoss(this.theta, this.features, this.labels, this.testStart, this.testEnd);
    this.lastAcc = classAccuracy(this.theta, this.features, this.labels, this.testStart, this.testEnd);
    this.history.push({ round: -1, loss: this.lastLoss, acc: this.lastAcc, accepted: false, delta: 0, ts: Date.now() });
    await this.publishSnapshot();
  }

  // Adam-on-scalar apply. Updates m, v, t with the winning scalar_g,
  // computes step_eff via bias-corrected moments, applies θ ← θ − step_eff·u.
  // The math here MUST be bit-equivalent to the worker's projection in
  // scripts/head-spsa-adam-verifier.mjs (computeStepEff there).
  private applySpsaStep(seed: number, scalar_g: number): void {
    this.adamT += 1;
    this.adamM = BETA1 * this.adamM + (1 - BETA1) * scalar_g;
    this.adamV = BETA2 * this.adamV + (1 - BETA2) * scalar_g * scalar_g;
    const mHat = this.adamM / (1 - Math.pow(BETA1, this.adamT));
    const vHat = this.adamV / (1 - Math.pow(BETA2, this.adamT));
    const stepEff = LR * mHat / (Math.sqrt(vHat) + EPS_ADAM);
    reconstructPerturbation(seed, this.uScratch);
    for (let i = 0; i < P; i++) this.theta[i] -= stepEff * this.uScratch[i];
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
        const quarantineHit = cumRate > 0.4 || winRate > 0.4 || longRate > 0.25;
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
      this.lastLoss = classLoss(this.theta, this.features, this.labels, this.testStart, this.testEnd);
      this.lastAcc = classAccuracy(this.theta, this.features, this.labels, this.testStart, this.testEnd);
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
        // Phase 39 — adaptive η. real_Δ on the held-out test set is the only
        // signal we trust (claimed_Δ comes from the worker's local shard and
        // is what the byzantine check above just judged). If real_Δ looks
        // like a real reduction, take larger steps; if it looks like a real
        // regression, shrink. Otherwise leave it alone (the noise band).
        // No post-apply adaptation needed — Adam moments already updated
        // inside applySpsaStep(). Kept the block here to mark the spot
        // structurally for future schedulers (e.g. LR warmup / cosine decay).
      }
      this.history.push({ round: this.round, loss: this.lastLoss, acc: this.lastAcc, accepted: !!apply, delta: appliedDelta, ts: Date.now() });
      if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
      this.round += 1;
      this.bestProposal = null;
      this.proposalsReceived = 0;
      advanced = true;
      this.broadcast({
        type: "advance",
        round: this.round,
        last_loss: this.lastLoss,
        last_acc: this.lastAcc,
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
      P, D, H, K,
      epsilon: EPSILON,
      lr: LR, beta1: BETA1, beta2: BETA2, eps_adam: EPS_ADAM,
      adam_m: this.adamM, adam_v: this.adamV, adam_t: this.adamT,
      train_n: TRAIN_N,
      test_n: this.testEnd - this.testStart,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: this.joined.size,
      last_loss: this.lastLoss,
      last_acc: this.lastAcc,
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
    // Effective current step magnitude (zero before any apply since m=v=0).
    const stepEffNow = this.adamT > 0
      ? LR * (this.adamM / (1 - Math.pow(BETA1, this.adamT)))
        / (Math.sqrt(this.adamV / (1 - Math.pow(BETA2, this.adamT))) + EPS_ADAM)
      : 0;
    return {
      round: this.round,
      P, D, H, K,
      class_names: CLASS_NAMES,
      epsilon: EPSILON,
      lr: LR, beta1: BETA1, beta2: BETA2, eps_adam: EPS_ADAM,
      adam_m: this.adamM, adam_v: this.adamV, adam_t: this.adamT,
      step_eff_now: stepEffNow,
      train_n: TRAIN_N,
      test_n: this.testEnd - this.testStart,
      worker_stats: workerReport,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: Array.from(this.joined),
      last_loss: this.lastLoss,
      last_acc: this.lastAcc,
      accepted: this.accepted,
      considered: this.considered,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      history: this.history.slice(-200),
      uptime_ms: Date.now() - this.bornAt,
    };
  }
}
