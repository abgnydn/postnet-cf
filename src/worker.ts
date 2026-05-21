/**
 * Postnet × Cloudflare — federated SGD coord.
 *
 * Browser workers compute the gradient of BCE loss on their local batch,
 * POST it (129 floats = ~520 B) to the coord. The coord averages all
 * gradients in the current round's pool and applies an SGD step.
 *
 * Architecture: 2 → H → 1 with ReLU + sigmoid, H=32, P=129.
 * Wavy boundary task: label = sin(2x) > y. Backprop solves this fast where
 * ES got stuck in the y-only basin.
 */
import { DurableObject } from "cloudflare:workers";

export interface Env {
  COORD: DurableObjectNamespace;
  ASSETS: Fetcher;
}

const H = 32;
const P = 4 * H + 1;          // 129 params
const TARGET_GRADIENTS = 2;   // average at least N worker gradients before stepping
const LR = 0.3;               // learning rate
const MOMENTUM = 0.9;         // helps push through plateaus the y-only basin sits in

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

function testLoss(theta: Float32Array): number {
  const rng = mulberry32(99999);
  let loss = 0;
  const N = 256;
  for (let i = 0; i < N; i++) {
    const x = (rng() - 0.5) * 4;
    const y = (rng() - 0.5) * 4;
    const label = Math.sin(2 * x) > y ? 1 : 0;
    const p = forward(theta, x, y);
    const eps = 1e-7;
    loss += -(label * Math.log(p + eps) + (1 - label) * Math.log(1 - p + eps));
  }
  return loss / N;
}

// --- Coord Durable Object ---
export class Coord extends DurableObject<Env> {
  private round = 0;
  private theta: Float32Array;
  private velocity: Float32Array;  // SGD-momentum buffer
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
    this.velocity = new Float32Array(P);
    this.lastLoss = testLoss(this.theta);
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
      const rng = mulberry32(Math.floor(Math.random() * 0xFFFFFFFF));
      this.theta = new Float32Array(P);
      for (let i = 0; i < P; i++) this.theta[i] = (rng() - 0.5) * 0.5;
      this.velocity = new Float32Array(P);
      this.round = 0;
      this.pool = [];
      this.history = [];
      this.lastLoss = testLoss(this.theta);
      this.history.push({ round: -1, loss: this.lastLoss, n: 0, ts: Date.now() });
      return Response.json({ ok: true });
    }
    return new Response("not found", { status: 404 });
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
      const avg = new Float32Array(P);
      for (const g of this.pool) {
        for (let i = 0; i < P; i++) avg[i] += g[i];
      }
      for (let i = 0; i < P; i++) avg[i] /= this.pool.length;
      // SGD with momentum: v ← β·v + g ; θ ← θ − lr·v
      const next = new Float32Array(P);
      for (let i = 0; i < P; i++) {
        this.velocity[i] = MOMENTUM * this.velocity[i] + avg[i];
        next[i] = this.theta[i] - LR * this.velocity[i];
      }
      this.theta = next;
      const used = this.pool.length;
      this.pool = [];
      this.lastLoss = testLoss(this.theta);
      this.history.push({ round: this.round, loss: this.lastLoss, n: used, ts: Date.now() });
      if (this.history.length > 500) this.history.splice(0, this.history.length - 500);
      this.round += 1;
      advanced = true;
    }

    return Response.json({
      round: this.round,
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
    if (url.pathname.startsWith("/api/")) {
      const id = env.COORD.idFromName("default");
      return env.COORD.get(id).fetch(request);
    }
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
