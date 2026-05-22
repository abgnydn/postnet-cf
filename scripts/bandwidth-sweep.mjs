// Bandwidth scaling sweep — what would the protocols cost at larger
// model sizes? Simulates the wire encoding for each phase across
// H = {32, 128, 512, 2048, 8192} (P = 4H + 1).
//
// This is a static analysis: no DO is spun up at the larger sizes
// (would require refactoring). Instead we construct a representative
// response payload for each protocol and measure JSON.stringify /
// binary serialization bytes — the bytes that would actually be on
// the wire if the coord were running at that size.

const FLIP_SIZE = 4;

function phase0AdamResponse(P) {
  // Federated Adam tick response (status quo for `/`)
  return {
    round: 100,
    theta: new Array(P).fill(0).map((_, i) => 0.1 - i * 1e-5),
    P,
    pool_size: 1,
    target: 2,
    joined: 3,
    last_loss: 0.2345,
    advanced: false,
  };
}

function phase1TournamentResponse(P) {
  // Phase 1 tournament tick response — flip uplink, full θ downlink
  return {
    round: 100,
    task: "wave",
    theta: new Array(P).fill(0).map((_, i) => 0.1 - i * 1e-5),
    P,
    flip_size: FLIP_SIZE,
    target: 2,
    proposals: 1,
    joined: 3,
    last_loss: 0.2345,
    accept_rate: 0.49,
    last_applied: {
      round: 99,
      indices: [4, 17, 42, 88],
      values: [0.012, -0.034, 0.21, -0.07],
    },
    advanced: true,
  };
}

function phase2TournamentResponse(P) {
  // Phase 2 tournament tick response — flip uplink, applied_since downlink (no θ)
  return {
    round: 100,
    task: "wave",
    P,
    flip_size: FLIP_SIZE,
    target: 2,
    proposals: 1,
    joined: 3,
    last_loss: 0.2345,
    accept_rate: 0.49,
    last_applied: {
      round: 99,
      indices: [4, 17, 42, 88],
      values: [0.012, -0.034, 0.21, -0.07],
    },
    applied_since: [{
      round: 99,
      indices: [4, 17, 42, 88],
      values: [0.012, -0.034, 0.21, -0.07],
    }],
    oldest_applied_round: 0,
    advanced: true,
  };
}

function bytes(obj) {
  return Buffer.byteLength(JSON.stringify(obj));
}

function bootstrapBinary(P) {
  return 8 + P * 4;  // 4-byte round + 4-byte P + P float32
}

function bootstrapJson(P) {
  // Phase 2 used JSON for bootstrap; Phase 3 uses binary
  return Buffer.byteLength(JSON.stringify({
    round: 0,
    task: "wave",
    P,
    theta: new Array(P).fill(0).map((_, i) => 0.1 - i * 1e-5),
  }));
}

function fmt(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

console.log("Bandwidth scaling across model sizes — wire-format bytes only.");
console.log("(Includes JSON envelope. Binary bootstrap = 8-byte header + P × float32.)\n");

console.log("| H | P | Adam ↓/tick | Phase 1 ↓/tick | Phase 2 ↓/tick | Bootstrap JSON | Bootstrap binary |");
console.log("|---|---|---|---|---|---|---|");

for (const H of [32, 128, 512, 2048, 8192]) {
  const P = 4 * H + 1;
  const adam = bytes(phase0AdamResponse(P));
  const p1 = bytes(phase1TournamentResponse(P));
  const p2 = bytes(phase2TournamentResponse(P));
  const bj = bootstrapJson(P);
  const bb = bootstrapBinary(P);
  console.log(`| ${H} | ${P.toLocaleString()} | ${fmt(adam)} | ${fmt(p1)} | ${fmt(p2)} | ${fmt(bj)} | ${fmt(bb)} |`);
}

// Hypothetical BitNet 2B (1.5 B ternary params, 1.58 bits/param)
console.log("\nProjected for BitNet 2B (1.5 B ternary params, 1.58 bits/param):");
const bitnet_P = 1_500_000_000;
const bitnet_ternary_bytes = Math.ceil(bitnet_P * 1.58 / 8);
console.log(`| BitNet 2B | ${bitnet_P.toLocaleString()} | ~${fmt(bitnet_ternary_bytes)} (overflows Worker 100 MB response cap) | same | ${fmt(bytes(phase2TournamentResponse(0)).valueOf())} (constant) | impossible via JSON | ${fmt(bitnet_ternary_bytes)} (one R2 fetch, range-readable) |`);

console.log("\nKey: every protocol's per-tick downlink dominates the long-run cost.");
console.log("Phase 0/1 grow linearly with P; Phase 2 stays constant after bootstrap.");
console.log("The bootstrap fetch is one-time per worker; in Phase 3 it's served from R2.");
