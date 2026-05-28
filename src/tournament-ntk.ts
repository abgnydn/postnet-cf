/**
 * Phase 40 next-2 — federated NTK-Mirror gate-controller training.
 *
 * Trains the K-vector of NTK-Mirror gate values (`raw[K]`) on top of a
 * frozen base model via the SPSA tournament from Phase 36/38. The static
 * gate-selection artifact (which (layer, channel) pairs to train) ships
 * as a public/data/ asset that BOTH this DO and the verifier/workers
 * fetch and parse.
 *
 * What this DO ships in next-2:
 *   ✓ load artifact from ASSETS in constructor → P = K (e.g. 5000)
 *   ✓ SPSA tournament (DeComFL-style, byte-identical to Phase 38)
 *   ✓ fixed η = 0.001 (Phase 38 style)
 *   ✓ sharded R2 snapshots of θ = raw[K]
 *   ✓ WS push for round advances
 *   ✓ applied_history for client reconciliation
 *   ✗ server-side test-loss        — DEFERRED to next-3 (CF Workers
 *   ✗ post-apply byzantine check     can't load 500M-param base model;
 *   ✗ adaptive η (Phase 39 logic)    needs a "loss oracle" companion)
 *
 * For next-3: bring back the Phase 39 sym-AIMD adaptation + byzantine
 * check once we wire in a loss oracle (either a Transformers.js bundle
 * inside the Worker, a trusted audit endpoint, or a per-round
 * worker-reported scalar with cross-validation).
 *
 * Wire format per proposal: 20 bytes (seed + scalar_g + claimed_Δ),
 * same as Phase 36+. The gate-selection artifact is fetched ONCE by
 * each worker out-of-band; only `raw[K]` updates over time.
 */
import { DurableObject } from "cloudflare:workers";
import { parseGateArtifact, type ParsedGateArtifact } from "./ntk-gate";

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
  TOURNAMENT_NTK: DurableObjectNamespace;
  ASSETS: Fetcher;
  SNAPSHOTS: R2Bucket;
}

// The artifact this DO trains against. Bake additional artifacts and
// flip this constant (or eventually parameterize the DO via idFromName).
const ARTIFACT_URL = "http://internal/data/qwen05b-math-gates-k5000.bin";

// Protocol constants.
const EPSILON = 0.005;            // SPSA perturbation magnitude
const ETA_INIT = 0.001;           // initial step size; ADAPTED per Phase 39 sym-AIMD
// Phase 39's symmetric AIMD multipliers — log-symmetric so η drifts based on
// the IMBALANCE of grow vs shrink events (typically ~70%-grow rate → η drifts up).
const ETA_UP = 1.05;
const ETA_DOWN = 1 / 1.05;
const ETA_MIN = 1e-5;
const ETA_MAX = 0.1;
// Phase 40 next-3: 1e-5 (tighter than Phase 39's 1e-4) because gate-controller
// per-round Δloss is tiny at init — gates start at zero so the model is
// effectively the base model and small θ steps barely shift loss. With this
// threshold the audit signal lifts above noise within ~10 rounds.
const ETA_DELTA_THRESH = 1e-5;
// TARGET = 1 for Phase 40 next-2/3 dev: the Python verifier (only Qwen-runner
// we have today) is heavy enough that running 2+ in parallel pressures the
// 16-GB Mac. Bump to 2 in a later phase once we have a lighter browser worker.
const TARGET_PROPOSALS = 1;
const SNAPSHOT_EVERY = 50;
const SNAPSHOT_KEY_PREFIX = "ntk/";
const SHARD_SIZE = 16 * 1024;
const FLOATS_PER_SHARD = SHARD_SIZE / 4;

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

// CRITICAL: must be bit-identical to scripts/ntk-verifier.py's
// _reconstruct_perturbation(). Same mulberry32 + Box-Muller scheme used
// in every postnet SPSA tournament since Phase 36.
function reconstructPerturbation(seed: number, out: Float32Array): void {
  const P = out.length;
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

export class TournamentNtk extends DurableObject<Env> {
  private round = 0;
  private theta!: Float32Array;            // raw[K] — trainable scalars
  private uScratch!: Float32Array;
  private artifact!: ParsedGateArtifact;
  private bestProposal: SpsaProposal | null = null;
  private proposalsReceived = 0;
  private joined = new Set<string>();
  private accepted = 0;
  private considered = 0;
  private history: { round: number; accepted: boolean; delta: number; loss?: number | null; ts: number }[] = [];
  private appliedHistory: SpsaAppliedFlip[] = [];
  private snapshotRound = 0;
  private snapshotShards: string[] = [];
  private subscribers = new Set<WebSocket>();
  private bornAt = Date.now();
  // Phase 40 next-3 — trusted-auditor loss oracle. Workers send
  // audit_loss_before in every tick (it's the loss they were going to
  // compute anyway for their claimed_Δ). Server uses it to drive Phase
  // 39 sym-AIMD η + byzantine real_Δ check on a one-round lag.
  private lastLoss: number | null = null;
  private savedLossBeforeApply: number | null = null;
  private pendingAudit: {
    round: number;
    workerId: string;
    claimedDelta: number;
  } | null = null;
  // Adaptive η (Phase 39 sym-AIMD), now driven by audited loss.
  private currentEta = ETA_INIT;
  private etaGrowEvents = 0;
  private etaShrinkEvents = 0;
  // Per-worker fraud tracking (Phase 9 / 14 / 31 tiered windows).
  private workerStats = new Map<string, {
    wins: number;
    frauds: number;
    lastWinRound: number;
    recent: number[];
    recent_long: number[];
  }>();

  constructor(state: DurableObjectState, env: Env) {
    super(state, env);
    state.blockConcurrencyWhile(async () => {
      const r = await env.ASSETS.fetch(ARTIFACT_URL);
      if (!r.ok) throw new Error(`gate artifact fetch ${r.status}`);
      const buf = await r.arrayBuffer();
      this.artifact = parseGateArtifact(buf);
      // theta = raw[K], starts from the artifact's raw_init (all zeros for
      // federated training — the swarm fills these in).
      this.theta = new Float32Array(this.artifact.raw);
      this.uScratch = new Float32Array(this.artifact.K);
      this.history.push({ round: -1, accepted: false, delta: 0, ts: this.bornAt });
      await this.publishSnapshot();
    });
  }

  private get P(): number {
    return this.artifact.K;
  }

  private async publishSnapshot(): Promise<void> {
    const numShards = Math.ceil(this.P / FLOATS_PER_SHARD);
    const shardKeys: string[] = [];
    const puts: Promise<unknown>[] = [];
    for (let k = 0; k < numShards; k++) {
      const floatStart = k * FLOATS_PER_SHARD;
      const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, this.P);
      const floatCount = floatEnd - floatStart;
      const headerSize = k === 0 ? 8 : 0;
      const buf = new ArrayBuffer(headerSize + floatCount * 4);
      const view = new DataView(buf);
      if (k === 0) {
        view.setUint32(0, this.round, true);
        view.setUint32(4, this.P, true);
      }
      new Float32Array(buf, headerSize, floatCount).set(this.theta.subarray(floatStart, floatEnd));
      const key = `${SNAPSHOT_KEY_PREFIX}r${this.round}/shard${k}.bin`;
      shardKeys.push(key);
      puts.push(this.env.SNAPSHOTS.put(key, buf, {
        httpMetadata: { contentType: "application/octet-stream" },
        customMetadata: {
          round: String(this.round),
          p: String(this.P),
          shard: String(k),
          total_shards: String(numShards),
          kind: "ntk",
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
    if (url.pathname === "/api/ntk/tick" && request.method === "POST") return this.tick(request);
    if (url.pathname === "/api/ntk/state") return Response.json(this.summary());
    if (url.pathname === "/api/ntk/snapshot") {
      const round = this.snapshotRound || this.round;
      const numShards = Math.ceil(this.P / FLOATS_PER_SHARD);
      const shards = [];
      for (let k = 0; k < numShards; k++) {
        const floatStart = k * FLOATS_PER_SHARD;
        const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, this.P);
        const floatCount = floatEnd - floatStart;
        const headerSize = k === 0 ? 8 : 0;
        shards.push({
          url: `/api/ntk/snapshot.bin?round=${round}&shard=${k}`,
          shard: k,
          float_start: floatStart,
          float_count: floatCount,
          bytes: headerSize + floatCount * 4,
        });
      }
      return Response.json({
        round,
        P: this.P,
        K: this.artifact.K,
        n_layers: this.artifact.nLayers,
        hidden_size: this.artifact.hiddenSize,
        max_log_gate: this.artifact.maxLogGate,
        model_id_hash: `0x${this.artifact.modelIdHash.toString(16).padStart(16, "0")}`,
        artifact_url: "/data/qwen05b-math-gates-k5000.bin",
        epsilon: EPSILON,
        eta: this.currentEta,
        eta_init: ETA_INIT,
        eta_grow_events: this.etaGrowEvents,
        eta_shrink_events: this.etaShrinkEvents,
        num_shards: numShards,
        shard_size_floats: FLOATS_PER_SHARD,
        shards,
        snapshot_bytes_total: 8 + this.P * 4,
      });
    }
    if (url.pathname === "/api/ntk/snapshot.bin") {
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
      const numShards = Math.ceil(this.P / FLOATS_PER_SHARD);
      const shardIdx = Math.max(0, Math.min(wantShard, numShards - 1));
      const floatStart = shardIdx * FLOATS_PER_SHARD;
      const floatEnd = Math.min(floatStart + FLOATS_PER_SHARD, this.P);
      const floatCount = floatEnd - floatStart;
      const headerSize = shardIdx === 0 ? 8 : 0;
      const buf = new ArrayBuffer(headerSize + floatCount * 4);
      const view = new DataView(buf);
      if (shardIdx === 0) {
        view.setUint32(0, this.round, true);
        view.setUint32(4, this.P, true);
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
    if (url.pathname === "/api/ntk/reset" && request.method === "POST") {
      await this.resetState();
      return Response.json({ ok: true });
    }
    if (url.pathname === "/api/ntk/ws") {
      const upgrade = request.headers.get("upgrade");
      if (upgrade !== "websocket") return new Response("expected websocket", { status: 426 });
      const pair = new WebSocketPair();
      const [client, server] = Object.values(pair);
      server.accept();
      this.subscribers.add(server);
      server.send(JSON.stringify({
        type: "hello",
        round: this.round,
        recent: this.appliedHistory.slice(-50),
        epsilon: EPSILON,
        eta: this.currentEta,
        eta_init: ETA_INIT,
        eta_grow_events: this.etaGrowEvents,
        eta_shrink_events: this.etaShrinkEvents,
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
    // Re-init raw to the artifact's raw_init (zeros). Keep loaded artifact.
    this.theta = new Float32Array(this.artifact.raw);
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
    this.lastLoss = null;
    this.savedLossBeforeApply = null;
    this.pendingAudit = null;
    this.currentEta = ETA_INIT;
    this.etaGrowEvents = 0;
    this.etaShrinkEvents = 0;
    this.workerStats.clear();
    this.history.push({ round: -1, accepted: false, delta: 0, ts: Date.now() });
    await this.publishSnapshot();
  }

  private applySpsaStep(seed: number, scalar_g: number): void {
    reconstructPerturbation(seed, this.uScratch);
    const k = this.currentEta * scalar_g;
    for (let i = 0; i < this.P; i++) this.theta[i] -= k * this.uScratch[i];
  }

  private async tick(request: Request): Promise<Response> {
    const body = await request.json<{
      worker_id: string;
      round?: number;
      seed?: number;
      scalar_g?: number;
      delta?: number;
      since_round?: number;
      // Phase 40 next-3: worker's local test loss BEFORE running this round's
      // trials. We trust it as the server's canonical lastLoss; it drives
      // byzantine real_Δ + sym-AIMD η adaptation on a one-round lag.
      audit_loss_before?: number;
    }>();
    if (!body.worker_id) return new Response("missing worker_id", { status: 400 });
    this.joined.add(body.worker_id);

    // ── PHASE 40 next-3: audit ingestion ────────────────────────────────────
    // If the worker reports a loss measurement, use it. If a previous round's
    // winner is awaiting audit, compute real_Δ NOW and run the byzantine check
    // + sym-AIMD η update on it.
    if (typeof body.audit_loss_before === "number") {
      const newLoss = body.audit_loss_before;
      if (this.pendingAudit && this.savedLossBeforeApply !== null) {
        const realGlobalDelta = newLoss - this.savedLossBeforeApply;
        const { workerId, claimedDelta } = this.pendingAudit;
        // Phase 9 / 14 / 31 byzantine math, unchanged from Phase 39.
        const stats = this.workerStats.get(workerId) ?? { wins: 0, frauds: 0, lastWinRound: -1, recent: [], recent_long: [] };
        stats.wins += 1;
        stats.lastWinRound = this.pendingAudit.round;
        const isFraud = realGlobalDelta > 1e-4 && claimedDelta < -1e-4;
        if (isFraud) stats.frauds += 1;
        stats.recent.push(isFraud ? 1 : 0);
        if (stats.recent.length > 20) stats.recent.shift();
        stats.recent_long.push(isFraud ? 1 : 0);
        if (stats.recent_long.length > 100) stats.recent_long.shift();
        this.workerStats.set(workerId, stats);
        // Phase 39 sym-AIMD η update — log-symmetric so η drifts on
        // imbalance of grow vs shrink events.
        if (realGlobalDelta < -ETA_DELTA_THRESH) {
          this.currentEta = Math.min(this.currentEta * ETA_UP, ETA_MAX);
          this.etaGrowEvents += 1;
        } else if (realGlobalDelta > ETA_DELTA_THRESH) {
          this.currentEta = Math.max(this.currentEta * ETA_DOWN, ETA_MIN);
          this.etaShrinkEvents += 1;
        }
        // The audit closed the loop for the prior winner; clear pending.
        this.pendingAudit = null;
        this.savedLossBeforeApply = null;
      }
      this.lastLoss = newLoss;
    }

    let accepted = false, rejected = false, quarantined = false;
    const hasProposal =
      typeof body.seed === "number" &&
      typeof body.scalar_g === "number" &&
      typeof body.delta === "number";
    if (hasProposal) {
      if (body.round === this.round) {
        // Phase 14+31 tiered quarantine, using stats accumulated above.
        const stats = this.workerStats.get(body.worker_id);
        const cumRate = stats && stats.wins >= 10 ? stats.frauds / stats.wins : 0;
        const recent = stats?.recent ?? [];
        const recentLong = stats?.recent_long ?? [];
        const winRate = recent.length >= 10
          ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
        const longRate = recentLong.length >= 30
          ? recentLong.reduce((a, b) => a + b, 0) / recentLong.length : 0;
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
        // Remember the loss BEFORE this apply so we can compute real_Δ once
        // the NEXT round's first audit arrives.
        this.savedLossBeforeApply = this.lastLoss;
        this.pendingAudit = {
          round: this.round,
          workerId: this.bestProposal!.worker_id,
          claimedDelta: this.bestProposal!.delta,
        };
      }
      this.history.push({
        round: this.round, accepted: !!apply, delta: appliedDelta,
        loss: this.lastLoss, ts: Date.now(),
      });
      if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
      this.round += 1;
      this.bestProposal = null;
      this.proposalsReceived = 0;
      advanced = true;
      this.broadcast({
        type: "advance",
        round: this.round,
        applied: appliedFlip,
        eta: this.currentEta,
      });
    }

    let appliedSince: SpsaAppliedFlip[] | null = null;
    if (typeof body.since_round === "number") {
      appliedSince = this.appliedHistory.filter(f => f.round >= body.since_round!);
    }
    const oldestAppliedRound = this.appliedHistory.length > 0 ? this.appliedHistory[0].round : null;

    return Response.json({
      round: this.round,
      P: this.P,
      K: this.artifact.K,
      n_layers: this.artifact.nLayers,
      hidden_size: this.artifact.hiddenSize,
      max_log_gate: this.artifact.maxLogGate,
      epsilon: EPSILON,
      eta: this.currentEta,
      eta_init: ETA_INIT,
      eta_grow_events: this.etaGrowEvents,
      eta_shrink_events: this.etaShrinkEvents,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: this.joined.size,
      last_applied: appliedFlip,
      applied_since: appliedSince,
      oldest_applied_round: oldestAppliedRound,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      accepted, rejected, quarantined,
      advanced,
      // null in next-2; reactivated in next-3 with a loss oracle.
      last_loss: this.lastLoss,
      pending_audit_round: this.pendingAudit ? this.pendingAudit.round : null,
    });
  }

  private summary() {
    // For convenience: also report a few cheap server-computable stats about
    // theta so the operator can see the swarm is doing SOMETHING.
    let sumAbs = 0, sumSq = 0, maxAbs = 0;
    for (let i = 0; i < this.P; i++) {
      const v = this.theta[i];
      const a = Math.abs(v);
      sumAbs += a;
      sumSq += v * v;
      if (a > maxAbs) maxAbs = a;
    }
    return {
      round: this.round,
      P: this.P,
      K: this.artifact.K,
      n_layers: this.artifact.nLayers,
      hidden_size: this.artifact.hiddenSize,
      max_log_gate: this.artifact.maxLogGate,
      model_id_hash: `0x${this.artifact.modelIdHash.toString(16).padStart(16, "0")}`,
      artifact_url: "/data/qwen05b-math-gates-k5000.bin",
      epsilon: EPSILON,
      eta: this.currentEta,
      eta_init: ETA_INIT,
      eta_grow_events: this.etaGrowEvents,
      eta_shrink_events: this.etaShrinkEvents,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: Array.from(this.joined),
      accepted: this.accepted,
      considered: this.considered,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      last_loss: this.lastLoss,
      pending_audit_round: this.pendingAudit ? this.pendingAudit.round : null,
      worker_stats: (() => {
        const r: Record<string, { wins: number; frauds: number; fraud_rate: number; lastWinRound: number }> = {};
        for (const [wid, st] of this.workerStats.entries()) {
          r[wid] = {
            wins: st.wins, frauds: st.frauds,
            fraud_rate: st.wins > 0 ? st.frauds / st.wins : 0,
            lastWinRound: st.lastWinRound,
          };
        }
        return r;
      })(),
      theta_stats: {
        mean_abs: sumAbs / this.P,
        l2_norm: Math.sqrt(sumSq),
        max_abs: maxAbs,
      },
      history: this.history.slice(-200),
      uptime_ms: Date.now() - this.bornAt,
    };
  }
}
