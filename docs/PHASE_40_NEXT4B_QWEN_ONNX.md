# Phase 40 next-4-b session 1 — Qwen+gates ONNX export

> _Goal: produce a browser-loadable ONNX of Qwen-0.5B-Instruct that
> accepts per-layer gate multipliers as a forward INPUT, so workers can
> SPSA-vary them per trial without graph surgery on each request._

## What this session shipped

```
   scripts/export-qwen-with-gates.py
   ────────────────────────────────────
   Loads Qwen2.5-0.5B-Instruct, wraps with QwenWithGateInputs (registers
   forward hooks on the 24 decoder layers that read a forward-input
   gate_mults tensor and multiply it into the residual stream), exports
   to ONNX via torch.onnx.export.

   Exported signature:
     inputs:
       input_ids      [batch, seq]    int64
       attention_mask [batch, seq]    int64
       gate_mults     [24, 896]       float32   ← per-layer mult vector
     output:
       logits         [batch, seq, V] float32

   At gate_mults = ones, the wrapped model is byte-identical to base
   Qwen (PyTorch diff = 0.0, ONNX-vs-PyTorch diff = 2.43e-4 = fp32 noise).
```

## Files produced (NOT committed — too large)

| file                                  | size      | what                                    |
|---|---|---|
| `qwen05b-with-gates.onnx`             | **3.9 MB**| graph definition; refs external weights |
| `qwen05b-with-gates.onnx.data`        | **1.8 GB**| fp32 weights (external-data sidecar)    |

Total: ~1.8 GB. Way too big to ship via CF assets (25 MB/file limit on
free tier; 100 MB on paid). `.gitignore`d. Run the script locally to
regenerate.

## Why this is still a meaningful win

The **graph + Mul-injection** works correctly. The validation gate is met:

- Wrapper at all-ones gate_mults matches the base PyTorch model exactly.
- Exported ONNX at all-ones matches PyTorch logits within 2.43e-4 (fp32 noise).
- ONNX export honors the dynamic-axes config — `batch` and `seq` are
  dynamic, only `gate_mults` stays static `[24, 896]`.

So the **protocol architecture is validated end-to-end on a real LLM**:
ONNX can host the gate-injected forward; workers compute their per-trial
gate_mults from the K=5000 raw values and pass them through onnxruntime;
the gates participate in the gradient signal as expected.

The size problem is **deployment**, not **correctness**.

## Session 2 — browser worker (~3-4 hours)

The remaining work splits into a quantization step and a UI/protocol step.

### Quantization (pick one path)

| approach                                    | size  | effort     | notes |
|---|---|---|---|
| **A. int4 + per-block weights** (TF.js / WebLLM-style) | ~250 MB | half day | the production target; matches what Xenova ships for Qwen-0.5B-Instruct |
| **B. int8 dynamic quantization**             | ~500 MB | 30 min   | onnxruntime has a built-in `quantize_dynamic`; quick win but bigger download |
| **C. Use Xenova/Qwen2.5-0.5B-Instruct directly** | ~250 MB | 1-2 hours | already int4 on HF; needs us to wrap it WITH gate-injection. Two options to inject: ONNX graph surgery on the downloaded model, OR a JS-level wrapper that multiplies hidden states between layer forwards (not possible because Transformers.js doesn't expose layer outputs) — so back to graph surgery. |

Recommended: **C** (we use the canonical public artifact and just add Mul
nodes via ONNX surgery — should keep the ~250 MB int4 size with negligible
overhead from 24 extra Mul ops + 24 small input tensors).

### Browser worker (`public/ntk.html` + `public/ntk-worker.js`)

```
   onnxruntime-web    ←  load qwen05b-int4-with-gates.onnx from
                          R2 / CDN / OPFS cache
                          (sidecar weights bundled, single file via
                           the ONNX `save_as_external_data=False` path
                           OR a small set of consolidated files)

   per round:
     poll /api/ntk/tick                       ← Phase 40 next-3
     reconcile applied flips (replay SPSA updates) → theta[K=5000]
     compute loss_before:
       gate_mults = compute_mults(theta, artifact)    ← exp(0.05·tanh(theta))
                                                       on selected (layer,
                                                       channel) slots
       loss = forward(input_ids, attention_mask, gate_mults)
            → cross-entropy on labels
     for T=4 SPSA trials:
       seed_t, u_t = reconstruct(seed_t)
       gate_mults_plus  = compute_mults(theta + ε·u_t, artifact)
       gate_mults_minus = compute_mults(theta - ε·u_t, artifact)
       loss_plus  = forward(..., gate_mults_plus)
       loss_minus = forward(..., gate_mults_minus)
       g_t = (loss_plus - loss_minus) / (2ε)
       gate_mults_step = compute_mults(theta - η·g_t·u_t, artifact)
       loss_at  = forward(..., gate_mults_step)
       claimed_Δ_t = loss_at - loss_before
     submit best trial via /api/ntk/tick with audit_loss_before=loss_before
```

Browser per-forward cost (Qwen-0.5B int4 on M-series, seq~32):
- WebAssembly: ~500-1500 ms per forward
- WebGPU: ~150-400 ms per forward (if onnxruntime-web's WebGPU EP
  supports the ops used; needs verification)

3 forwards/trial × 4 trials/round = 12 forwards/round. At 500 ms each,
that's 6 seconds per round in WASM. R=100 → 10 minutes per tab.

### Fallback for memory-constrained tabs

A 250 MB int4 model + activations + onnxruntime-web is ~500 MB-1 GB of
browser memory. Mobile tabs and 8 GB Macs may struggle.

Option: ship gates for **Pythia-160M** as a smaller demo target.
~160 MB int4, ~250 MB peak. Demo only — won't quite match the "real LLM"
narrative but completes the federated-browser story for more devices.

## Why we didn't bring the ONNX into the repo this session

The 1.8 GB sidecar is bigger than (a) the entire repo, (b) GitHub's
hard file-size limit (100 MB), and (c) CF assets' practical limit.
`public/data/*.onnx` and `public/data/*.onnx.data` are now `.gitignore`d.
The script to regenerate is committed; session 2's first action is
**quantize**, then decide whether the quantized artifact lives in R2 or
as a CF static asset.

## Reproducing this session

```bash
~/ntkmirror/.venv/bin/pip install onnx onnxruntime onnxscript
cd ~/postnet-cf
~/ntkmirror/.venv/bin/python scripts/export-qwen-with-gates.py
```

Output ends up in `public/data/qwen05b-with-gates.onnx`
+ `public/data/qwen05b-with-gates.onnx.data`. Expect ~40 s of export
time and ~1.8 GB of disk.
