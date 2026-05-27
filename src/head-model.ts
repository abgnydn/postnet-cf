/**
 * Phase 38 — shared head-classifier model.
 *
 * 2-layer MLP over pre-computed MiniLM-L6-v2 features:
 *
 *     features (D=384)  →  fc1 + bias  (D × H)
 *                       →  ReLU        (H)
 *                       →  fc2 + bias  (H × K)
 *                       →  logits      (K)
 *
 * With H=128, K=4 → P = 49 796 (~50K) — solidly in SPSA-wins territory
 * per Phase 37's empirical crossover.
 *
 * Imported by both src/tournament-head-flip.ts and src/tournament-head-spsa.ts
 * so the flip/SPSA pair compare on byte-identical forward/loss math.
 */

export const D = 384;     // MiniLM-L6-v2 feature dim
export const H = 128;     // hidden width
export const K = 4;       // num classes (AG News: World/Sports/Business/Sci-Tech)

export const P_FC1 = D * H;            // 49152
export const P_B1 = H;                 // 128
export const P_FC2 = H * K;            // 512
export const P_B2 = K;                 // 4
export const P = P_FC1 + P_B1 + P_FC2 + P_B2;   // 49796

export const FC1_OFF = 0;
export const B1_OFF = FC1_OFF + P_FC1;
export const FC2_OFF = B1_OFF + P_B1;
export const B2_OFF = FC2_OFF + P_FC2;

export const CLASS_NAMES = ["World", "Sports", "Business", "Sci-Tech"] as const;

// One forward pass for a single feature vector x of length D.
// outLogits is filled in place with K logits (no softmax — caller can softmax
// or take argmax depending on need).
export function forward(theta: Float32Array, x: Float32Array, outLogits: Float32Array): void {
  // h = bias_h + x @ FC1   (D → H)
  const h = new Float32Array(H);
  for (let j = 0; j < H; j++) h[j] = theta[B1_OFF + j];
  for (let i = 0; i < D; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = FC1_OFF + i * H;
    for (let j = 0; j < H; j++) h[j] += xi * theta[row + j];
  }
  // ReLU
  for (let j = 0; j < H; j++) if (h[j] < 0) h[j] = 0;
  // logits = bias_k + h @ FC2   (H → K)
  for (let k = 0; k < K; k++) outLogits[k] = theta[B2_OFF + k];
  for (let j = 0; j < H; j++) {
    const hj = h[j];
    if (hj === 0) continue;
    const row = FC2_OFF + j * K;
    for (let k = 0; k < K; k++) outLogits[k] += hj * theta[row + k];
  }
}

// Mean cross-entropy loss over examples [start, end) in (features, labels).
// features is laid out [N × D] row-major; labels is [N] uint8.
export function classLoss(
  theta: Float32Array,
  features: Float32Array,
  labels: Uint8Array,
  start: number,
  end: number,
): number {
  const logits = new Float32Array(K);
  const x = new Float32Array(D);
  let loss = 0;
  for (let n = start; n < end; n++) {
    for (let i = 0; i < D; i++) x[i] = features[n * D + i];
    forward(theta, x, logits);
    let mx = logits[0];
    for (let k = 1; k < K; k++) if (logits[k] > mx) mx = logits[k];
    let sum = 0;
    for (let k = 0; k < K; k++) sum += Math.exp(logits[k] - mx);
    const tgt = labels[n];
    loss += -(logits[tgt] - mx - Math.log(sum + 1e-7));
  }
  return loss / Math.max(end - start, 1);
}

// Top-1 accuracy over examples [start, end).
export function classAccuracy(
  theta: Float32Array,
  features: Float32Array,
  labels: Uint8Array,
  start: number,
  end: number,
): number {
  const logits = new Float32Array(K);
  const x = new Float32Array(D);
  let correct = 0;
  for (let n = start; n < end; n++) {
    for (let i = 0; i < D; i++) x[i] = features[n * D + i];
    forward(theta, x, logits);
    let arg = 0;
    for (let k = 1; k < K; k++) if (logits[k] > logits[arg]) arg = k;
    if (arg === labels[n]) correct++;
  }
  return correct / Math.max(end - start, 1);
}

// He-style init: each row of FC1/FC2 ~ N(0, 2/fan_in). Biases at zero.
// Deterministic from `seed` so worker and server start identical.
export function initTheta(seed: number): Float32Array {
  const theta = new Float32Array(P);
  let t = seed >>> 0;
  const rng = () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const sd1 = Math.sqrt(2 / D);
  for (let i = 0; i < P_FC1; i++) theta[FC1_OFF + i] = gauss() * sd1;
  // biases zero (already)
  const sd2 = Math.sqrt(2 / H);
  for (let i = 0; i < P_FC2; i++) theta[FC2_OFF + i] = gauss() * sd2;
  return theta;
}

// Binary dataset file layout written by scripts/extract-agnews-features.mjs:
//   [uint32 N][uint32 D][uint32 K][uint32 _reserved]
//   [N × D × float32 features]
//   [N × uint8 labels]
export interface ParsedDataset {
  N: number;
  D: number;
  K: number;
  features: Float32Array;
  labels: Uint8Array;
}

export function parseDataset(buf: ArrayBuffer): ParsedDataset {
  const view = new DataView(buf);
  const N = view.getUint32(0, true);
  const fileD = view.getUint32(4, true);
  const fileK = view.getUint32(8, true);
  if (fileD !== D) throw new Error(`feature dim mismatch: file=${fileD} code=${D}`);
  if (fileK !== K) throw new Error(`class count mismatch: file=${fileK} code=${K}`);
  const headerBytes = 16;
  const features = new Float32Array(buf, headerBytes, N * D).slice();
  const labels = new Uint8Array(buf, headerBytes + N * D * 4, N).slice();
  return { N, D, K, features, labels };
}
