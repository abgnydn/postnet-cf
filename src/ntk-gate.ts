/**
 * Phase 40 — NTK-Mirror gate-apply hook for postnet-cf.
 *
 * Loads the binary gate-selection artifact emitted by
 * `scripts/extract-ntk-gates.py` and exposes:
 *
 *   parseGateArtifact(buf)                → parsed header + index arrays
 *   buildGateIndex(parsed)                → per-layer (channels, slot) lookup
 *   applyGatesToLayer(h, BT, C, ...)      → in-place h[:,:,c] *= exp(s_{l,c})
 *
 * Wire-format identical to the Python writer (little-endian throughout):
 *
 *   HEADER (32 B):
 *     [ 0:  4]  bytes "NTKG"          magic
 *     [ 4:  8]  u32   version = 1
 *     [ 8: 12]  u32   K               number of selected gates
 *     [12: 16]  u32   n_layers
 *     [16: 20]  u32   hidden_size
 *     [20: 24]  f32   max_log_gate
 *     [24: 32]  u64   model_id_hash   (FNV-1a 64 of model name)
 *
 *   BODY:
 *     K × u16  layer_indices
 *     K × u16  channel_indices
 *     K × f32  raw_init                (zeros for federated training)
 *
 * Total: 32 + 8*K bytes. K=5000 → 40 032 B.
 *
 * Imported by:
 *   - src/tournament-ntk.ts                  (the federated DO)
 *   - public/ntk-worker.js                   (browser worker; via a
 *                                              wasm/JS bundle bridge)
 *   - scripts/ntk-verifier.mjs               (Node verifier)
 */

export const NTK_MAGIC = "NTKG";
export const NTK_VERSION = 1;
export const NTK_HEADER_SIZE = 32;

export interface ParsedGateArtifact {
  /** Number of selected gates. P_controller = K. */
  K: number;
  nLayers: number;
  hiddenSize: number;
  maxLogGate: number;
  /** FNV-1a 64-bit hash of the model name string (use to detect mismatches). */
  modelIdHash: bigint;
  /** Length K, values in [0, nLayers). */
  layerIndices: Uint16Array;
  /** Length K, values in [0, hiddenSize). */
  channelIndices: Uint16Array;
  /** Length K. Server-canonical trainable scalars. Updated via SPSA. */
  raw: Float32Array;
}

export class GateArtifactParseError extends Error {}

/**
 * Parse the binary artifact. Throws on shape mismatch or version skew.
 * Holds no references into `buf` after returning — caller can free.
 */
export function parseGateArtifact(buf: ArrayBuffer): ParsedGateArtifact {
  if (buf.byteLength < NTK_HEADER_SIZE) {
    throw new GateArtifactParseError(
      `artifact too small: ${buf.byteLength} < header ${NTK_HEADER_SIZE}`
    );
  }
  const view = new DataView(buf);

  // Magic check — read as ASCII to match the Python writer (bytes b"NTKG"
  // = 0x4E 0x54 0x4B 0x47, regardless of native endianness).
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
  );
  if (magic !== NTK_MAGIC) {
    throw new GateArtifactParseError(
      `bad magic: got '${magic}', expected '${NTK_MAGIC}'`
    );
  }

  const version = view.getUint32(4, true);
  if (version !== NTK_VERSION) {
    throw new GateArtifactParseError(
      `unsupported artifact version ${version} (this code: ${NTK_VERSION})`
    );
  }

  const K = view.getUint32(8, true);
  const nLayers = view.getUint32(12, true);
  const hiddenSize = view.getUint32(16, true);
  const maxLogGate = view.getFloat32(20, true);
  const modelIdHash = view.getBigUint64(24, true);

  const expectedTotal = NTK_HEADER_SIZE + K * 2 + K * 2 + K * 4;
  if (buf.byteLength < expectedTotal) {
    throw new GateArtifactParseError(
      `artifact truncated: ${buf.byteLength} bytes, expected ${expectedTotal} for K=${K}`
    );
  }

  // Each typed-array view shares the underlying buffer initially; .slice() copies
  // so the caller can drop the original buffer if they want.
  let off = NTK_HEADER_SIZE;
  const layerIndices = new Uint16Array(buf, off, K).slice();
  off += K * 2;
  const channelIndices = new Uint16Array(buf, off, K).slice();
  off += K * 2;
  const raw = new Float32Array(buf, off, K).slice();

  // Sanity (cheap, worth it for debug clarity).
  for (let i = 0; i < K; i++) {
    if (layerIndices[i] >= nLayers) {
      throw new GateArtifactParseError(
        `layer_indices[${i}] = ${layerIndices[i]} >= n_layers ${nLayers}`
      );
    }
    if (channelIndices[i] >= hiddenSize) {
      throw new GateArtifactParseError(
        `channel_indices[${i}] = ${channelIndices[i]} >= hidden_size ${hiddenSize}`
      );
    }
  }

  return { K, nLayers, hiddenSize, maxLogGate, modelIdHash, layerIndices, channelIndices, raw };
}

/**
 * Stable 64-bit FNV-1a of a UTF-8 string. Matches the Python `fnv1a_64()` in
 * scripts/extract-ntk-gates.py — use to verify the worker is running against
 * the same base model the gates were selected for.
 */
export function fnv1a64(s: string): bigint {
  let h = 0xCBF29CE484222325n;
  const enc = new TextEncoder().encode(s);
  const PRIME = 0x100000001B3n;
  const MASK = 0xFFFFFFFFFFFFFFFFn;
  for (const b of enc) {
    h ^= BigInt(b);
    h = (h * PRIME) & MASK;
  }
  return h;
}

/**
 * Per-layer index. For a given layer L, returns:
 *   channels[layer L]      — Uint16Array of selected channels (length n_L)
 *   slotIndices[layer L]   — Uint32Array, same length: slot j into `raw`
 *
 * Built once at startup; the per-token application path is then a tight loop
 * over the selected channels only — no scan of all K gates per layer.
 */
export interface PerLayerGateIndex {
  /** channels[layer] = Uint16Array of selected channels in that layer. */
  channels: Uint16Array[];
  /** slotIndices[layer][j] = slot j of `raw` corresponding to channels[layer][j]. */
  slotIndices: Uint32Array[];
}

export function buildGateIndex(parsed: ParsedGateArtifact): PerLayerGateIndex {
  const { K, nLayers, layerIndices, channelIndices } = parsed;
  // Two-pass: count, allocate, fill.
  const counts = new Uint32Array(nLayers);
  for (let i = 0; i < K; i++) counts[layerIndices[i]]++;
  const channels: Uint16Array[] = new Array(nLayers);
  const slotIndices: Uint32Array[] = new Array(nLayers);
  const cursors = new Uint32Array(nLayers);
  for (let l = 0; l < nLayers; l++) {
    channels[l] = new Uint16Array(counts[l]);
    slotIndices[l] = new Uint32Array(counts[l]);
  }
  for (let i = 0; i < K; i++) {
    const l = layerIndices[i];
    const j = cursors[l]++;
    channels[l][j] = channelIndices[i];
    slotIndices[l][j] = i;
  }
  return { channels, slotIndices };
}

/**
 * Apply gates to one decoder layer's hidden state IN PLACE.
 *
 * Shape contract:
 *   h is a Float32Array (or compatible) representing one layer's output,
 *   logical shape [BT, C] in row-major C-contiguous layout where:
 *     BT = batch × tokens (flattened)
 *     C  = hiddenSize
 *
 * For each gate j at this layer:
 *   s_j = max_log_gate * tanh(raw[slot_j])     (matches Python controller.py)
 *   m_j = exp(s_j)
 *   h[i, channel_j] *= m_j   for i in [0, BT)
 *
 * NOTE: `raw` here is the LIVE trainable vector — both the federated SPSA
 * apply path and per-trial probe-step path call this with their respective
 * theta. The function does NOT internally cache m_j across calls because
 * raw[] changes every step.
 */
export function applyGatesToLayer(
  h: Float32Array,
  BT: number,
  C: number,
  channels: Uint16Array,
  slotIndices: Uint32Array,
  raw: Float32Array,
  maxLogGate: number,
): void {
  if (h.length !== BT * C) {
    throw new Error(`h length ${h.length} != BT*C ${BT}*${C}`);
  }
  const nGatesHere = channels.length;
  for (let j = 0; j < nGatesHere; j++) {
    const c = channels[j];
    const r = raw[slotIndices[j]];
    // s_j = max_log_gate * tanh(r); m_j = exp(s_j).
    // tanh clamps to (-1, +1) so |s_j| < max_log_gate always.
    const s = maxLogGate * Math.tanh(r);
    const m = Math.exp(s);
    // Multiply column c in place.
    for (let i = 0; i < BT; i++) {
      h[i * C + c] *= m;
    }
  }
}

/**
 * Convenience: apply gates to a whole stack of layer outputs at once.
 * `hStack[l]` is layer l's hidden state (Float32Array of length BT*C).
 * Layers with no selected gates are skipped (no allocation, no loop).
 */
export function applyGatesToStack(
  hStack: (Float32Array | null)[],
  BT: number,
  C: number,
  index: PerLayerGateIndex,
  raw: Float32Array,
  maxLogGate: number,
): void {
  const nLayers = index.channels.length;
  if (hStack.length !== nLayers) {
    throw new Error(`hStack length ${hStack.length} != n_layers ${nLayers}`);
  }
  for (let l = 0; l < nLayers; l++) {
    const channels = index.channels[l];
    if (channels.length === 0) continue;
    const h = hStack[l];
    if (h == null) continue;
    applyGatesToLayer(h, BT, C, channels, index.slotIndices[l], raw, maxLogGate);
  }
}
