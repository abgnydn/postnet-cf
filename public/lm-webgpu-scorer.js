/**
 * Phase 35 — WebGPU char-LM scorer.
 *
 * Drop-in replacement for the JS textLoss() in lm-worker.js. Same math
 * (context-2 MLP: embed V×E → fc1 2E×H + bias → ReLU → fc2 H×V + bias →
 * softmax CE-loss), batched so all positions in the shard run in a single
 * compute dispatch. Returns a Promise<number> with the mean per-char loss.
 *
 * On hardware without WebGPU, init() returns null and the worker falls
 * back to the JS path. The two paths converge at parity within ~1e-5
 * float-precision noise (different summation order on the GPU).
 */

const WGSL = /* wgsl */`
const V: u32 = 27u;
const E: u32 = 16u;
const HID: u32 = 32u;
const P_EMBED: u32 = 432u;
const FC1_OFF: u32 = 432u;
const B1_OFF: u32 = 1456u;
const FC2_OFF: u32 = 1488u;
const B2_OFF: u32 = 2352u;

@group(0) @binding(0) var<storage, read>       theta:  array<f32>;
@group(0) @binding(1) var<storage, read>       codes:  array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;
// params.x = start, params.y = end, params.z = count = end - start

@compute @workgroup_size(64)
fn textLoss(@builtin(global_invocation_id) gid: vec3<u32>) {
  let count: u32 = params.z;
  if (gid.x >= count) { return; }
  let i: u32 = params.x + gid.x;

  let prevPrev: u32 = codes[i - 2u];
  let prev:     u32 = codes[i - 1u];
  let tgt:      u32 = codes[i];

  // h = bias_h
  var h: array<f32, 32>;
  for (var j: u32 = 0u; j < HID; j = j + 1u) { h[j] = theta[B1_OFF + j]; }

  // h += embed[prevPrev] @ FC1[0..E, :]
  for (var k: u32 = 0u; k < E; k = k + 1u) {
    let xi: f32 = theta[prevPrev * E + k];
    let row: u32 = FC1_OFF + k * HID;
    for (var j: u32 = 0u; j < HID; j = j + 1u) { h[j] = h[j] + xi * theta[row + j]; }
  }
  // h += embed[prev] @ FC1[E..2E, :]
  for (var k: u32 = 0u; k < E; k = k + 1u) {
    let xi: f32 = theta[prev * E + k];
    let row: u32 = FC1_OFF + (E + k) * HID;
    for (var j: u32 = 0u; j < HID; j = j + 1u) { h[j] = h[j] + xi * theta[row + j]; }
  }
  // ReLU
  for (var j: u32 = 0u; j < HID; j = j + 1u) { if (h[j] < 0.0) { h[j] = 0.0; } }

  // logits = bias_v + h @ FC2
  var logits: array<f32, 32>;
  for (var v: u32 = 0u; v < V; v = v + 1u) { logits[v] = theta[B2_OFF + v]; }
  for (var j: u32 = 0u; j < HID; j = j + 1u) {
    let hj: f32 = h[j];
    let row: u32 = FC2_OFF + j * V;
    for (var v: u32 = 0u; v < V; v = v + 1u) { logits[v] = logits[v] + hj * theta[row + v]; }
  }

  // softmax-CE
  var mx: f32 = logits[0];
  for (var v: u32 = 1u; v < V; v = v + 1u) { if (logits[v] > mx) { mx = logits[v]; } }
  var sum: f32 = 0.0;
  for (var v: u32 = 0u; v < V; v = v + 1u) { sum = sum + exp(logits[v] - mx); }
  losses[gid.x] = -(logits[tgt] - mx - log(sum + 1e-7));
}
`;

const P = 2379;
const CODES_MAX = 1024;     // codes buffer headroom; current TEXT is 340 chars
const LOSSES_MAX = 1024;    // per-position losses headroom

export async function createWebGPUScorer(codes) {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  let adapter, device;
  try {
    adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    device = await adapter.requestDevice();
  } catch {
    return null;
  }

  const module = device.createShaderModule({ code: WGSL });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "textLoss" },
  });

  const thetaBuf = device.createBuffer({
    size: P * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const codesBuf = device.createBuffer({
    size: CODES_MAX * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const lossesBuf = device.createBuffer({
    size: LOSSES_MAX * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuf = device.createBuffer({
    size: LOSSES_MAX * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const paramsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // codes is static — upload once. Pack u8s into u32s (one code per word; wastes
  // 75% of the buffer but at 340 chars that's 1 KB versus the worth of indexing
  // gymnastics in WGSL).
  const codesU32 = new Uint32Array(CODES_MAX);
  for (let i = 0; i < codes.length && i < CODES_MAX; i++) codesU32[i] = codes[i];
  device.queue.writeBuffer(codesBuf, 0, codesU32);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: thetaBuf } },
      { binding: 1, resource: { buffer: codesBuf } },
      { binding: 2, resource: { buffer: lossesBuf } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ],
  });

  const paramsScratch = new Uint32Array(4);

  async function textLoss(theta, shardStart, shardEnd) {
    const start = Math.max(shardStart != null ? shardStart : 0, 2);
    const end = shardEnd != null ? shardEnd : codes.length;
    const count = Math.max(end - start, 0);
    if (count === 0) return 0;
    if (count > LOSSES_MAX) throw new Error(`shard too large: ${count} > ${LOSSES_MAX}`);

    device.queue.writeBuffer(thetaBuf, 0, theta.buffer, theta.byteOffset, theta.byteLength);
    paramsScratch[0] = start;
    paramsScratch[1] = end;
    paramsScratch[2] = count;
    paramsScratch[3] = 0;
    device.queue.writeBuffer(paramsBuf, 0, paramsScratch);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    enc.copyBufferToBuffer(lossesBuf, 0, stagingBuf, 0, count * 4);
    device.queue.submit([enc.finish()]);

    await stagingBuf.mapAsync(GPUMapMode.READ, 0, count * 4);
    const view = new Float32Array(stagingBuf.getMappedRange(0, count * 4).slice(0));
    stagingBuf.unmap();

    let sum = 0;
    for (let i = 0; i < count; i++) sum += view[i];
    return sum / count;
  }

  function dispose() {
    thetaBuf.destroy();
    codesBuf.destroy();
    lossesBuf.destroy();
    stagingBuf.destroy();
    paramsBuf.destroy();
    device.destroy();
  }

  return { textLoss, dispose, backend: "webgpu" };
}
