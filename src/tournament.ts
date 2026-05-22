/**
 * Phase 1 of the fusedx integration: tournament / flip-and-accept protocol.
 *
 * Each round:
 *   1. Coord broadcasts current θ
 *   2. Each worker generates K random "flip" proposals locally
 *      (a flip = perturb FLIP_SIZE random parameters by Gaussian noise),
 *      scores each on its private batch, keeps its single best
 *   3. Worker POSTs that single best proposal to coord
 *   4. Coord picks the proposal with the largest negative Δloss across
 *      *all* workers in the round and applies it to θ
 *   5. Round advances; broadcast new θ; repeat
 *
 * Bandwidth per worker per round: ~FLIP_SIZE × 8 bytes (indices + values),
 * independent of model size. The mismatch shape from the integration doc.
 */
import { DurableObject } from "cloudflare:workers";

export interface Env {
  COORD: DurableObjectNamespace;
  TOURNAMENT: DurableObjectNamespace;
  ASSETS: Fetcher;
}

const H = 32;
const P = 4 * H + 1;
const TARGET_PROPOSALS = 2;
const FLIP_SIZE = 4;     // how many params each flip touches

type Task = "wave" | "circle" | "xor";
const TASKS: Task[] = ["wave", "circle", "xor"];

function trueLabel(task: Task, x: number, y: number): number {
  switch (task) {
    case "wave":   return Math.sin(2 * x) > y ? 1 : 0;
    case "circle": return (x * x + y * y) < 1 ? 1 : 0;
    case "xor":    return ((x > 0) !== (y > 0)) ? 1 : 0;
  }
}

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

function forward(theta: Float32Array, x: number, y: number): number {
  let z = theta[4 * H];
  const w2Off = 3 * H;
  const b1Off = 2 * H;
  for (let i = 0; i < H; i++) {
    let h = theta[i * 2] * x + theta[i * 2 + 1] * y + theta[b1Off + i];
    if (h < 0) h = 0;
    z += theta[w2Off + i] * h;
  }
  return 1 / (1 + Math.exp(-z));
}

function testLoss(theta: Float32Array, task: Task): number {
  const rng = mulberry32(99999);
  let loss = 0;
  const N = 256;
  for (let i = 0; i < N; i++) {
    const x = (rng() - 0.5) * 4;
    const y = (rng() - 0.5) * 4;
    const label = trueLabel(task, x, y);
    const p = forward(theta, x, y);
    const eps = 1e-7;
    loss += -(label * Math.log(p + eps) + (1 - label) * Math.log(1 - p + eps));
  }
  return loss / N;
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

export class Tournament extends DurableObject<Env> {
  private round = 0;
  private task: Task = "wave";
  private theta: Float32Array;
  private bestProposal: Proposal | null = null;
  private proposalsReceived = 0;
  private joined = new Set<string>();
  private lastLoss = -1;
  private accepted = 0;        // count of proposals actually applied
  private considered = 0;      // count of proposals received total
  private history: { round: number; loss: number; accepted: boolean; delta: number; ts: number }[] = [];
  private appliedHistory: AppliedFlip[] = [];
  private bornAt = Date.now();

  constructor(state: DurableObjectState, env: Env) {
    super(state, env);
    const rng = mulberry32(7);
    this.theta = new Float32Array(P);
    for (let i = 0; i < P; i++) this.theta[i] = (rng() - 0.5) * 0.5;
    this.lastLoss = testLoss(this.theta, this.task);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: this.bornAt });
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/tournament/tick" && request.method === "POST") {
      return this.tick(request);
    }
    if (url.pathname === "/api/tournament/state") {
      return Response.json(this.summary());
    }
    if (url.pathname === "/api/tournament/snapshot") {
      // Bootstrap endpoint — full θ for new workers or workers who drifted.
      // Phase 3 will move this payload to R2 (with a versioned URL).
      return Response.json({
        round: this.round,
        task: this.task,
        P,
        theta: Array.from(this.theta),
      });
    }
    if (url.pathname === "/api/tournament/reset" && request.method === "POST") {
      this.resetState();
      return Response.json({ ok: true, task: this.task });
    }
    if (url.pathname === "/api/tournament/set_task" && request.method === "POST") {
      const body = await request.json<{ task?: string }>();
      const t = body.task as Task;
      if (!TASKS.includes(t)) return new Response("invalid task", { status: 400 });
      this.task = t;
      this.resetState();
      return Response.json({ ok: true, task: this.task });
    }
    return new Response("not found", { status: 404 });
  }

  private resetState() {
    const rng = mulberry32(Math.floor(Math.random() * 0xFFFFFFFF));
    this.theta = new Float32Array(P);
    for (let i = 0; i < P; i++) this.theta[i] = (rng() - 0.5) * 0.5;
    this.round = 0;
    this.bestProposal = null;
    this.proposalsReceived = 0;
    this.joined.clear();
    this.accepted = 0;
    this.considered = 0;
    this.history = [];
    this.appliedHistory = [];
    this.lastLoss = testLoss(this.theta, this.task);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: Date.now() });
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

    let accepted = false;
    let rejected = false;
    if (Array.isArray(body.indices) && Array.isArray(body.values) && typeof body.delta === "number") {
      if (body.round === this.round && body.indices.length === body.values.length) {
        this.considered += 1;
        // Keep the best proposal seen this round (most negative delta)
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
      const applied = this.bestProposal && this.bestProposal.delta < 0;
      if (applied) {
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
      }
      this.lastLoss = testLoss(this.theta, this.task);
      this.history.push({
        round: this.round,
        loss: this.lastLoss,
        accepted: !!applied,
        delta: appliedDelta,
        ts: Date.now(),
      });
      if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
      this.round += 1;
      this.bestProposal = null;
      this.proposalsReceived = 0;
      advanced = true;
    }

    // Phase 2: delta-only response. Worker sends since_round (its localRound);
    // server replies with every applied flip since then so the worker can
    // catch up on rounds advanced by other workers' submissions.
    let appliedSince: AppliedFlip[] | null = null;
    if (typeof body.since_round === "number") {
      appliedSince = this.appliedHistory.filter(f => f.round >= body.since_round!);
    }
    const oldestAppliedRound = this.appliedHistory.length > 0
      ? this.appliedHistory[0].round
      : null;
    return Response.json({
      round: this.round,
      task: this.task,
      P,
      flip_size: FLIP_SIZE,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: this.joined.size,
      last_loss: this.lastLoss,
      last_applied: appliedFlip,
      applied_since: appliedSince,
      oldest_applied_round: oldestAppliedRound,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      accepted,
      rejected,
      advanced,
    });
  }

  private summary() {
    return {
      round: this.round,
      task: this.task,
      tasks: TASKS,
      P,
      flip_size: FLIP_SIZE,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: Array.from(this.joined),
      last_loss: this.lastLoss,
      accepted: this.accepted,
      considered: this.considered,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      history: this.history.slice(-200),
      theta: Array.from(this.theta),
      uptime_ms: Date.now() - this.bornAt,
    };
  }
}
