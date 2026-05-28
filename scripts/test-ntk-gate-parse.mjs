// Phase 40 — parity test for src/ntk-gate.ts parser vs Python writer.
// Verifies that public/data/qwen05b-math-gates-k5000.bin (written by
// scripts/extract-ntk-gates.py) round-trips through TS parseGateArtifact()
// with the expected shape and values.

import { readFileSync } from "node:fs";
import {
  parseGateArtifact,
  buildGateIndex,
  fnv1a64,
  applyGatesToLayer,
} from "../src/ntk-gate.ts";

const ARTIFACT = "public/data/qwen05b-math-gates-k5000.bin";
const buf = readFileSync(ARTIFACT);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const parsed = parseGateArtifact(ab);

console.log("parsed:", {
  K: parsed.K,
  nLayers: parsed.nLayers,
  hiddenSize: parsed.hiddenSize,
  maxLogGate: parsed.maxLogGate,
  modelIdHash: `0x${parsed.modelIdHash.toString(16).padStart(16, "0")}`,
});

let fail = 0;
function check(name, cond) {
  console.log(`  ${cond ? "✓" : "✗"}  ${name}`);
  if (!cond) fail++;
}

check("magic + version parsed", true);    // would have thrown otherwise
check("K == 5000",                parsed.K === 5000);
check("nLayers == 24",            parsed.nLayers === 24);
check("hiddenSize == 896",        parsed.hiddenSize === 896);
check("maxLogGate ≈ 0.05",        Math.abs(parsed.maxLogGate - 0.05) < 1e-6);
check("layerIndices length K",    parsed.layerIndices.length === parsed.K);
check("channelIndices length K",  parsed.channelIndices.length === parsed.K);
check("raw length K",             parsed.raw.length === parsed.K);
check("raw is all zeros (init)",  parsed.raw.every(v => v === 0));

// Verify per-layer histogram matches Python output:
// {0: 264, 1: 273, 2: 321, 3: 265, 4: 273, 5: 283, 6: 307, 7: 297, 8: 331,
//  9: 282, 10: 303, 11: 219, 12: 220, 13: 170, 14: 169, 15: 155, 16: 110,
//  17: 97, 18: 90, 19: 94, 20: 87, 21: 74, 22: 68, 23: 248}
const EXPECTED_COUNTS = [
  264, 273, 321, 265, 273, 283, 307, 297, 331, 282, 303, 219, 220, 170,
  169, 155, 110, 97, 90, 94, 87, 74, 68, 248,
];
const counts = new Array(parsed.nLayers).fill(0);
for (let i = 0; i < parsed.K; i++) counts[parsed.layerIndices[i]]++;
const histOK = counts.every((c, i) => c === EXPECTED_COUNTS[i]);
check("per-layer histogram matches Python writer", histOK);
if (!histOK) {
  console.log("    expected:", EXPECTED_COUNTS);
  console.log("    got:     ", counts);
}

// Model ID hash sanity: FNV-1a 64 of "Qwen/Qwen2.5-0.5B-Instruct" in TS
// must match Python's value (0xe52b123dcf9ef82c from extraction run).
const expectedHash = 0xe52b123dcf9ef82cn;
const tsHash = fnv1a64("Qwen/Qwen2.5-0.5B-Instruct");
check("fnv1a64('Qwen/Qwen2.5-0.5B-Instruct') matches Python",
      tsHash === expectedHash && parsed.modelIdHash === expectedHash);
if (tsHash !== expectedHash) {
  console.log(`    expected: 0x${expectedHash.toString(16)}`);
  console.log(`    ts:       0x${tsHash.toString(16)}`);
  console.log(`    artifact: 0x${parsed.modelIdHash.toString(16)}`);
}

// Channel range
let cMin = parsed.channelIndices[0], cMax = parsed.channelIndices[0];
for (let i = 1; i < parsed.K; i++) {
  if (parsed.channelIndices[i] < cMin) cMin = parsed.channelIndices[i];
  if (parsed.channelIndices[i] > cMax) cMax = parsed.channelIndices[i];
}
check("channelIndices in valid range",
      cMin >= 0 && cMax < parsed.hiddenSize);
console.log(`  (channel range: [${cMin}, ${cMax}])`);

// Build per-layer index + sanity
const idx = buildGateIndex(parsed);
let totalSelected = 0;
for (const ch of idx.channels) totalSelected += ch.length;
check("buildGateIndex total channels == K", totalSelected === parsed.K);
check("buildGateIndex layer 23 has 248 gates", idx.channels[23].length === 248);

// Application math: with raw=0, tanh(0)=0, exp(0)=1 → gates are NO-OPS at init.
// Verify by applying to a known-value buffer and confirming nothing changed.
const BT = 4, C = parsed.hiddenSize;
const h = new Float32Array(BT * C);
for (let i = 0; i < h.length; i++) h[i] = (i % 7) + 1.5;     // arbitrary non-zero
const hBefore = h.slice();
applyGatesToLayer(h, BT, C, idx.channels[23], idx.slotIndices[23], parsed.raw, parsed.maxLogGate);
let drift = 0;
for (let i = 0; i < h.length; i++) drift += Math.abs(h[i] - hBefore[i]);
check("applyGatesToLayer is NO-OP at raw=0 (init)", drift === 0);

// Application math: bump raw[slot 0] to 2.0 (saturates tanh). Should multiply
// channel idx.channels[23][0] across all BT rows by exp(0.05 * tanh(2)) ≈ exp(0.0481).
parsed.raw[idx.slotIndices[23][0]] = 2.0;
applyGatesToLayer(h, BT, C, idx.channels[23], idx.slotIndices[23], parsed.raw, parsed.maxLogGate);
const expectedMult = Math.exp(parsed.maxLogGate * Math.tanh(2.0));
const c0 = idx.channels[23][0];
const r0 = h[0 * C + c0] / hBefore[0 * C + c0];
check(`applyGatesToLayer with raw=2.0 → mult ≈ exp(0.05·tanh(2)) = ${expectedMult.toFixed(6)}`,
      Math.abs(r0 - expectedMult) < 1e-5);
console.log(`    measured mult on channel ${c0}: ${r0.toFixed(6)}`);

// Reset for cleanliness
parsed.raw[idx.slotIndices[23][0]] = 0;

console.log();
if (fail === 0) {
  console.log("ALL CHECKS PASSED");
  process.exit(0);
} else {
  console.log(`${fail} CHECK(S) FAILED`);
  process.exit(1);
}
