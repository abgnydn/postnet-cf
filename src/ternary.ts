/**
 * Phase 4 — ternary-weight tournament.
 *
 * Same architecture (2 → H → 1 MLP, H=32, P=129) and same flip-and-accept
 * coordination as the float-weight Tournament, but every weight is
 * constrained to {-1, 0, +1} (storage) × a single learned scale S.
 * The effective weight at index i is theta[i] = sign[i] * scale.
 *
 * Why: BitNet b1.58 uses ternary weights. Proving the postnet-cf
 * protocol drives ternary weights on the toy task is the substrate
 * test for swapping in a real BitNet via fused-lora as Phase 5.
 *
 * Snapshot wire format (packed):
 *   [uint32 round][uint32 P][float32 scale][ceil(P*2/8) bytes packed signs]
 * Each ternary value occupies 2 bits: 00 = 0, 01 = +1, 10 = -1.
 * Snapshot size at P = 1.5B ternary params: 4 + 4 + 4 + 375 MB ≈ 375 MB
 * (vs 6 GB for float32 — 16× smaller).
 */
import { DurableObject } from "cloudflare:workers";

export interface Env {
  COORD: DurableObjectNamespace;
  TOURNAMENT: DurableObjectNamespace;
  TERNARY: DurableObjectNamespace;
  ASSETS: Fetcher;
  SNAPSHOTS: R2Bucket;
}

const H = 32;
const P = 4 * H + 1;
const TARGET_PROPOSALS = 2;
const FLIP_SIZE = 6;     // ternary needs more flips per round (only 3 values)
const SNAPSHOT_EVERY = 50;
const SNAPSHOT_KEY_PREFIX = "ternary/";

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

function forward(sign: Int8Array, scale: number, x: number, y: number): number {
  // theta[i] = sign[i] * scale
  let z = sign[4 * H] * scale;
  const w2Off = 3 * H;
  const b1Off = 2 * H;
  for (let i = 0; i < H; i++) {
    let h = (sign[i * 2] * x + sign[i * 2 + 1] * y) * scale + sign[b1Off + i] * scale;
    if (h < 0) h = 0;
    z += sign[w2Off + i] * scale * h;
  }
  return 1 / (1 + Math.exp(-z));
}

function testLoss(sign: Int8Array, scale: number, task: Task): number {
  const rng = mulberry32(99999);
  let loss = 0;
  const N = 256;
  for (let i = 0; i < N; i++) {
    const x = (rng() - 0.5) * 4;
    const y = (rng() - 0.5) * 4;
    const label = trueLabel(task, x, y);
    const p = forward(sign, scale, x, y);
    const eps = 1e-7;
    loss += -(label * Math.log(p + eps) + (1 - label) * Math.log(1 - p + eps));
  }
  return loss / N;
}

// Pack P ternary values (each ∈ {-1,0,+1}) into ceil(P*2/8) bytes.
// Encoding: 00 = 0, 01 = +1, 10 = -1.
function packTernary(sign: Int8Array): Uint8Array {
  const bytes = Math.ceil(P * 2 / 8);
  const out = new Uint8Array(bytes);
  for (let i = 0; i < P; i++) {
    const code = sign[i] === 0 ? 0 : sign[i] > 0 ? 1 : 2;
    const byteIdx = (i * 2) >> 3;
    const bitOff = (i * 2) & 7;
    out[byteIdx] |= code << bitOff;
  }
  return out;
}

interface Proposal {
  worker_id: string;
  indices: number[];
  values: number[];   // each ∈ {-1, 0, +1}
  delta: number;
}

interface AppliedFlip {
  round: number;
  indices: number[];
  values: number[];
}

export class Ternary extends DurableObject<Env> {
  private round = 0;
  private task: Task = "wave";
  private sign: Int8Array;       // P × {-1, 0, +1}
  private scale: number;          // single global scale factor
  private bestProposal: Proposal | null = null;
  private proposalsReceived = 0;
  private joined = new Set<string>();
  private lastLoss = -1;
  private accepted = 0;
  private considered = 0;
  private history: { round: number; loss: number; accepted: boolean; delta: number; ts: number }[] = [];
  private appliedHistory: AppliedFlip[] = [];
  private snapshotRound = 0;
  private snapshotKey = "";
  private bornAt = Date.now();

  constructor(state: DurableObjectState, env: Env) {
    super(state, env);
    this.sign = new Int8Array(P);
    this.scale = 0.5;
    const rng = mulberry32(42);
    for (let i = 0; i < P; i++) {
      const r = rng();
      this.sign[i] = r < 0.33 ? -1 : r < 0.66 ? 0 : 1;
    }
    this.lastLoss = testLoss(this.sign, this.scale, this.task);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: this.bornAt });
    state.blockConcurrencyWhile(async () => {
      await this.publishSnapshot();
    });
  }

  private async publishSnapshot(): Promise<void> {
    const key = `${SNAPSHOT_KEY_PREFIX}r${this.round}.bin`;
    const packed = packTernary(this.sign);
    // [uint32 round][uint32 P][float32 scale][packed signs...]
    const buf = new ArrayBuffer(12 + packed.length);
    const view = new DataView(buf);
    view.setUint32(0, this.round, true);
    view.setUint32(4, P, true);
    view.setFloat32(8, this.scale, true);
    new Uint8Array(buf, 12).set(packed);
    try {
      await this.env.SNAPSHOTS.put(key, buf, {
        httpMetadata: { contentType: "application/octet-stream" },
        customMetadata: { round: String(this.round), task: this.task, p: String(P), kind: "ternary" },
      });
      this.snapshotRound = this.round;
      this.snapshotKey = key;
    } catch {}
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/ternary/tick" && request.method === "POST") {
      return this.tick(request);
    }
    if (url.pathname === "/api/ternary/state") {
      return Response.json(this.summary());
    }
    if (url.pathname === "/api/ternary/snapshot") {
      const bytes = 12 + Math.ceil(P * 2 / 8);
      return Response.json({
        round: this.snapshotRound || this.round,
        task: this.task,
        P,
        scale: this.scale,
        snapshot_url: `/api/ternary/snapshot.bin?round=${this.snapshotRound || this.round}`,
        snapshot_bytes: bytes,
      });
    }
    if (url.pathname === "/api/ternary/snapshot.bin") {
      const wantRound = parseInt(url.searchParams.get("round") || "-1");
      if (this.snapshotKey && (wantRound < 0 || wantRound === this.snapshotRound)) {
        const obj = await this.env.SNAPSHOTS.get(this.snapshotKey);
        if (obj) {
          return new Response(obj.body, {
            headers: {
              "content-type": "application/octet-stream",
              "x-snapshot-round": String(this.snapshotRound),
              "x-snapshot-source": "r2",
              "x-snapshot-kind": "ternary",
            },
          });
        }
      }
      const packed = packTernary(this.sign);
      const buf = new ArrayBuffer(12 + packed.length);
      const view = new DataView(buf);
      view.setUint32(0, this.round, true);
      view.setUint32(4, P, true);
      view.setFloat32(8, this.scale, true);
      new Uint8Array(buf, 12).set(packed);
      return new Response(buf, {
        headers: {
          "content-type": "application/octet-stream",
          "x-snapshot-round": String(this.round),
          "x-snapshot-source": "memory",
          "x-snapshot-kind": "ternary",
        },
      });
    }
    if (url.pathname === "/api/ternary/reset" && request.method === "POST") {
      await this.resetState();
      return Response.json({ ok: true, task: this.task });
    }
    if (url.pathname === "/api/ternary/set_task" && request.method === "POST") {
      const body = await request.json<{ task?: string }>();
      const t = body.task as Task;
      if (!TASKS.includes(t)) return new Response("invalid task", { status: 400 });
      this.task = t;
      await this.resetState();
      return Response.json({ ok: true, task: this.task });
    }
    return new Response("not found", { status: 404 });
  }

  private async resetState() {
    this.sign = new Int8Array(P);
    this.scale = 0.5;
    const rng = mulberry32(Math.floor(Math.random() * 0xFFFFFFFF));
    for (let i = 0; i < P; i++) {
      const r = rng();
      this.sign[i] = r < 0.33 ? -1 : r < 0.66 ? 0 : 1;
    }
    this.round = 0;
    this.bestProposal = null;
    this.proposalsReceived = 0;
    this.joined.clear();
    this.accepted = 0;
    this.considered = 0;
    this.history = [];
    this.appliedHistory = [];
    this.snapshotRound = 0;
    this.snapshotKey = "";
    this.lastLoss = testLoss(this.sign, this.scale, this.task);
    this.history.push({ round: -1, loss: this.lastLoss, accepted: false, delta: 0, ts: Date.now() });
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

    let accepted = false;
    let rejected = false;
    if (Array.isArray(body.indices) && Array.isArray(body.values) && typeof body.delta === "number") {
      // Sanity: ternary values must be in {-1, 0, +1}
      const valid = body.values.every(v => v === -1 || v === 0 || v === 1);
      if (body.round === this.round && body.indices.length === body.values.length && valid) {
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
          if (idx >= 0 && idx < P) this.sign[idx] = this.bestProposal!.values[i] as -1 | 0 | 1;
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
      this.lastLoss = testLoss(this.sign, this.scale, this.task);
      this.history.push({
        round: this.round,
        loss: this.lastLoss,
        accepted: !!apply,
        delta: appliedDelta,
        ts: Date.now(),
      });
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
    const oldestAppliedRound = this.appliedHistory.length > 0
      ? this.appliedHistory[0].round
      : null;

    return Response.json({
      round: this.round,
      task: this.task,
      P,
      scale: this.scale,
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
      scale: this.scale,
      flip_size: FLIP_SIZE,
      target: TARGET_PROPOSALS,
      proposals: this.proposalsReceived,
      joined: Array.from(this.joined),
      last_loss: this.lastLoss,
      accepted: this.accepted,
      considered: this.considered,
      accept_rate: this.considered > 0 ? this.accepted / this.considered : 0,
      history: this.history.slice(-200),
      sign: Array.from(this.sign),  // for boundary preview (small at P=129)
      uptime_ms: Date.now() - this.bornAt,
    };
  }
}
