/**
 * Phase 38 — head-classifier tournament, flip-and-accept variant.
 *
 * Federated training of a 2-layer MLP head over pre-computed MiniLM
 * features on AG News (4 classes, 100 examples, P=49 796).
 *
 *   train split:   first 75 examples (further sharded by worker)
 *   test split:    last  25 examples (server-side only, for byzantine
 *                  real_Δ check and reported test loss/accuracy)
 *
 * Same protocol shape as TournamentLM: {indices[6], values[6], delta}
 * proposals, post-apply real_Δ fraud check, R2 sharded snapshots, WS push.
 * The point of this DO is to be the *control* against the SPSA variant —
 * Phase 37 predicts SPSA wins at this P.
 */
import { DurableObject } from "cloudflare:workers";
import {
  P, D, H, K, FC1_OFF, B1_OFF, FC2_OFF, B2_OFF,
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
  ASSETS: Fetcher;
  SNAPSHOTS: R2Bucket;
}

const TRAIN_N = 75;
const TARGET_PROPOSALS = 2;
const FLIP_SIZE = 6;
const FLIP_SIGMA = 0.05;     // smaller than 0.15 for char-LM — MiniLM features have ~1.5 std
const SNAPSHOT_EVERY = 50;
const SNAPSHOT_KEY_PREFIX = "head-flip/";
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

export class TournamentHeadFlip extends DurableObject<Env> {
  private round = 0;
  private theta!: Float32Array;
  private features!: Float32Array;
  private labels!: Uint8Array;
  private N = 0;
  private testStart = 0;
  private testEnd = 0;
  private bestProposal: Proposal | null = null;
  private proposalsReceived = 0;
  private joined = new Set<string>();
  private lastLoss = -1;
  private lastAcc = -1;
  private accepted = 0;
  private considered = 0;
  private history: { round: number; loss: number; acc: number; accepted: boolean; delta: number; ts: number }[] = [];
  private appliedHistory: AppliedFlip[] = [];
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
          kind: "head-flip",
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
    if (url.pathname === "/api/head-flip/tick" && request.method === "POST") return this.tick(request);
    if (url.pathname === "/api/head-flip/state") return Response.json(this.summary());
    if (url.pathname === "/api/head-flip/snapshot") {
      const round = this.snapshotRound || this.round;
      const numShards = Math.ceil(P / FLOATS_PER_SHARD);
      const shards = [];
      for (let k = 0; k < numShards; k++) {
        const floatStart = k * FLOATS_PER_SHARD;
        const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, P);
        const floatCount = floatEnd - floatStart;
        const headerSize = k === 0 ? 8 : 0;
        shards.push({
          url: `/api/head-flip/snapshot.bin?round=${round}&shard=${k}`,
          shard: k,
          float_start: floatStart,
          float_count: floatCount,
          bytes: headerSize + floatCount * 4,
        });
      }
      return Response.json({
        round, P, D, H, K,
        train_n: TRAIN_N,
        test_n: this.testEnd - this.testStart,
        num_shards: numShards,
        shard_size_floats: FLOATS_PER_SHARD,
        shards,
        snapshot_bytes_total: 8 + P * 4,
      });
    }
    if (url.pathname === "/api/head-flip/snapshot.bin") {
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
    if (url.pathname === "/api/head-flip/reset" && request.method === "POST") {
      await this.resetState();
      return Response.json({ ok: true });
    }
    if (url.pathname === "/api/head-flip/ws") {
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
    // (the matching change lives in TournamentHeadSpsa.resetState).
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
    this.lastLoss = classLoss(this.theta, this.features, this.labels, this.testStart, this.testEnd);
    this.lastAcc = classAccuracy(this.theta, this.features, this.labels, this.testStart, this.testEnd);
    this.history.push({ round: -1, loss: this.lastLoss, acc: this.lastAcc, accepted: false, delta: 0, ts: Date.now() });
    await this.publishSnapshot();
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

    let appliedSince: AppliedFlip[] | null = null;
    if (typeof body.since_round === "number") {
      appliedSince = this.appliedHistory.filter(f => f.round >= body.since_round!);
    }
    const oldestAppliedRound = this.appliedHistory.length > 0 ? this.appliedHistory[0].round : null;

    return Response.json({
      round: this.round,
      P, D, H, K,
      train_n: TRAIN_N,
      test_n: this.testEnd - this.testStart,
      flip_size: FLIP_SIZE,
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
    return {
      round: this.round,
      P, D, H, K,
      class_names: CLASS_NAMES,
      train_n: TRAIN_N,
      test_n: this.testEnd - this.testStart,
      worker_stats: workerReport,
      flip_size: FLIP_SIZE,
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
