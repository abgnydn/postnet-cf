/**
 * Postnet × Cloudflare — federated Adam coord.
 *
 * Browser workers compute the gradient of BCE loss on their local batch,
 * POST it (129 floats = ~520 B) to the coord. The coord averages all
 * gradients in the current round's pool and applies an Adam step.
 *
 * Adam was chosen after SGD+momentum delivered a noisy loss curve — Adam's
 * per-parameter adaptive scaling produces smoother convergence on the
 * same task, same substrate.
 *
 * Architecture: 2 → H → 1 with ReLU + sigmoid, H=32, P=129.
 * Wavy boundary task: label = sin(2x) > y.
 */
import { DurableObject } from "cloudflare:workers";
export { Tournament } from "./tournament";

export interface Env {
  COORD: DurableObjectNamespace;
  TOURNAMENT: DurableObjectNamespace;
  ASSETS: Fetcher;
}

const H = 32;
const P = 4 * H + 1;            // 129 params
const TARGET_GRADIENTS = 2;     // average at least N worker gradients before stepping
const LR = 0.05;                // Adam typically wants smaller LR than SGD+momentum
const ADAM_B1 = 0.9;
const ADAM_B2 = 0.999;
const ADAM_EPS = 1e-8;

type Task = "wave" | "circle" | "xor";
const TASKS: Task[] = ["wave", "circle", "xor"];

function trueLabel(task: Task, x: number, y: number): number {
  switch (task) {
    case "wave":   return Math.sin(2 * x) > y ? 1 : 0;
    case "circle": return (x * x + y * y) < 1 ? 1 : 0;
    case "xor":    return ((x > 0) !== (y > 0)) ? 1 : 0;
  }
}

// --- Deterministic PRNG for the held-out test set ---
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

// --- Forward pass (also used to evaluate test loss server-side) ---
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

// --- Coord Durable Object ---
export class Coord extends DurableObject<Env> {
  private round = 0;
  private task: Task = "wave";
  private theta: Float32Array;
  private adamM: Float32Array;     // Adam 1st moment
  private adamV: Float32Array;     // Adam 2nd moment
  private adamStep = 0;            // step counter for bias correction
  private pool: Float32Array[] = [];
  private joined = new Set<string>();
  private lastLoss = -1;
  private history: { round: number; loss: number; n: number; ts: number }[] = [];
  private bornAt = Date.now();

  constructor(state: DurableObjectState, env: Env) {
    super(state, env);
    const rng = mulberry32(42);
    this.theta = new Float32Array(P);
    for (let i = 0; i < P; i++) this.theta[i] = (rng() - 0.5) * 0.5;
    this.adamM = new Float32Array(P);
    this.adamV = new Float32Array(P);
    this.lastLoss = testLoss(this.theta, this.task);
    this.history.push({ round: -1, loss: this.lastLoss, n: 0, ts: this.bornAt });
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/tick" && request.method === "POST") {
      return this.tick(request);
    }
    if (url.pathname === "/api/state") {
      return Response.json(this.summary());
    }
    if (url.pathname === "/api/reset" && request.method === "POST") {
      this.resetState();
      return Response.json({ ok: true, task: this.task });
    }
    if (url.pathname === "/api/set_task" && request.method === "POST") {
      const body = await request.json<{ task?: string }>();
      const t = body.task as Task;
      if (!TASKS.includes(t)) {
        return new Response("invalid task", { status: 400 });
      }
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
    this.adamM = new Float32Array(P);
    this.adamV = new Float32Array(P);
    this.adamStep = 0;
    this.round = 0;
    this.pool = [];
    this.joined.clear();
    this.history = [];
    this.lastLoss = testLoss(this.theta, this.task);
    this.history.push({ round: -1, loss: this.lastLoss, n: 0, ts: Date.now() });
  }

  private async tick(request: Request): Promise<Response> {
    const body = await request.json<{
      worker_id: string;
      round?: number;
      gradient?: number[];
    }>();
    if (!body.worker_id) return new Response("missing worker_id", { status: 400 });
    this.joined.add(body.worker_id);

    let accepted = false;
    let rejected = false;
    if (body.gradient && body.gradient.length === P) {
      if (body.round === this.round) {
        this.pool.push(Float32Array.from(body.gradient));
        accepted = true;
      } else {
        rejected = true;
      }
    }

    let advanced = false;
    if (this.pool.length >= TARGET_GRADIENTS) {
      // Average all gradients in the pool
      const g = new Float32Array(P);
      for (const gi of this.pool) {
        for (let i = 0; i < P; i++) g[i] += gi[i];
      }
      for (let i = 0; i < P; i++) g[i] /= this.pool.length;
      // Adam: m ← β1·m + (1-β1)·g ; v ← β2·v + (1-β2)·g² ; bias-correct ; θ ← θ - lr·m̂/(√v̂+ε)
      this.adamStep += 1;
      const bc1 = 1 - Math.pow(ADAM_B1, this.adamStep);
      const bc2 = 1 - Math.pow(ADAM_B2, this.adamStep);
      const next = new Float32Array(P);
      for (let i = 0; i < P; i++) {
        this.adamM[i] = ADAM_B1 * this.adamM[i] + (1 - ADAM_B1) * g[i];
        this.adamV[i] = ADAM_B2 * this.adamV[i] + (1 - ADAM_B2) * g[i] * g[i];
        const mHat = this.adamM[i] / bc1;
        const vHat = this.adamV[i] / bc2;
        next[i] = this.theta[i] - LR * mHat / (Math.sqrt(vHat) + ADAM_EPS);
      }
      this.theta = next;
      const used = this.pool.length;
      this.pool = [];
      this.lastLoss = testLoss(this.theta, this.task);
      this.history.push({ round: this.round, loss: this.lastLoss, n: used, ts: Date.now() });
      if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
      this.round += 1;
      advanced = true;
    }

    return Response.json({
      round: this.round,
      task: this.task,
      theta: Array.from(this.theta),
      P,
      lr: LR,
      pool_size: this.pool.length,
      target: TARGET_GRADIENTS,
      joined: this.joined.size,
      last_loss: this.lastLoss,
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
      lr: LR,
      target: TARGET_GRADIENTS,
      pool_size: this.pool.length,
      joined: Array.from(this.joined),
      last_loss: this.lastLoss,
      history: this.history.slice(-200),
      theta: Array.from(this.theta),
      uptime_ms: Date.now() - this.bornAt,
    };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.startsWith("/api/tournament/")) {
      const id = env.TOURNAMENT.idFromName("default");
      return env.TOURNAMENT.get(id).fetch(request);
    }
    if (url.pathname.startsWith("/api/")) {
      const id = env.COORD.idFromName("default");
      return env.COORD.get(id).fetch(request);
    }
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
