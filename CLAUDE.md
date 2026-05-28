# ~/postnet-cf — CLAUDE.md

postnet-cf: federated tournament protocol on Cloudflare Workers. 34 phases shipped + deployed 2026-05-26; phases 35–37 (WebGPU substrate, SPSA fusion, scaling crossover) added 2026-05-26/27 — see "Resume here" for state.

## 🎯 Resume here (on "continue")

_Updated: 2026-05-28 (Phases 39 + 39b shipped; sym-AIMD remains canonical; Phase 40 plan = NTK-Mirror integration)_

**Where we are.** Phase 39 + 39b shipped: a head-to-head of three η-adaptation rules on the Phase 38 head-classifier (P=49 796). **Symmetric AIMD (Phase 39, ×1.05 / ×1/1.05)** won decisively: at R=90, loss 1.40→1.12 and acc 32%→56% (vs fixed-η's 1.30 / 40% at R=100). **Adam-on-scalar (Phase 39b, MEAZO-faithful)** with textbook hyperparams underperforms — its step normalization caps the effective step magnitude near `lr`, while sym-AIMD lets η drift up unboundedly. MEAZO's "single global scalar matters most" framing is empirically supported; the SHAPE of the adaptation matters more than its sophistication. Sym-AIMD is the canonical Phase 39 algorithm. See `docs/PHASE_39_ADAPTIVE_ETA.md` and `docs/PHASE_39B_ADAM_ON_SCALAR.md`.

**Files staged but uncommitted (as of 2026-05-28):**

- `src/tournament-spsa-lm.ts` + `src/tournament-spsa-lm-big.ts` — SPSA DO (P=2 379 and P=31 707) [Phase 36-37]
- `src/tournament-lm-big.ts` — flip-and-accept at the big scale (control) [Phase 37]
- `src/tournament-head-flip.ts` + `src/tournament-head-spsa.ts` + `src/head-model.ts` — head-classifier DOs + shared model [Phase 38]
- `src/tournament-head-spsa-adaptive.ts` + `scripts/head-spsa-adaptive-verifier.mjs` + `docs/PHASE_39_ADAPTIVE_ETA.md` — adaptive-η variant [Phase 39]
- `src/tournament-head-spsa-adam.ts` + `scripts/head-spsa-adam-verifier.mjs` + `docs/PHASE_39B_ADAM_ON_SCALAR.md` — Adam-on-scalar variant [Phase 39b]
- `src/worker.ts` + `wrangler.jsonc` — bindings + migrations (now 8 tournament DOs total)
- `public/lm-webgpu-scorer.js` + `public/lm-parity.html` — WebGPU substrate (Phase 35; parity ✓, 0.55× speed on tiny model)
- `public/spsa-lm-worker.js` + `public/spsa-lm.html` — browser SPSA worker + demo page
- `public/data/agnews-mini.bin` — 100-example MiniLM features (~154 KB) [Phase 38]
- `scripts/extract-agnews-features.mjs` — one-shot feature extraction via @huggingface/transformers
- `scripts/spsa-verifier.mjs` + `scripts/{lm,spsa}-big-verifier.mjs` + `scripts/head-{flip,spsa}-verifier.mjs`
- `docs/PHASE_37_SCALING.md` + `docs/PHASE_38_HEAD.md` — empirical writeups
- `README.md` — phase table extended (35/36/37/38)
- `package.json` + `package-lock.json` — added `@huggingface/transformers` devDep

**Highest-impact moves remaining (in priority order; reorganized after the NTK-Mirror discovery):**

1. **Phase 40 — NTK-Mirror federated controller training (BIG SWING).** [github.com/leochlon/ntkmirror, MIT, Cambridge / Hassana Labs / Leon Chlon, May 2026, 184★ in 4 days]. Their controller is a sparse set of signed log-gates on residual-stream channels: `h'[:,:,c] *= exp(s_{l,c})`, top-K=5 000 selected by `|dL/ds|`. P=5 000 sits PRECISELY in our SPSA sweet spot (Phase 37 crossover at ~30K) and is forward-pass-only by construction. The fusion:
   - One-time central setup: score+select top-K (layer, channel) pairs on a representative corpus.
   - Workers run forward through a frozen base (Phi-3 via neuropulse's WGSL engine), SPSA-estimate the K-vector of gate values, submit (seed, scalar_g, claimed_Δ).
   - Tournament + apply via Phase 39's adaptive η. Byte-identical 20-byte wire.
   - **Bonus:** controllers ADD in log space → federated training across shards/tasks produces a LIBRARY of composable skill modules (~20 KB each). NTK-Mirror's "persistent memory" feature becomes a federated artifact.
   - This is the paper-grade target. ~1 week. Bridge files: `~/neuropulse/src/engine/inference.ts` + 11 WGSL kernels + `src/ntkmirror/controller.py` from the ntkmirror repo (port gate-apply to TS).

2. **Phase 39b — Adam-on-scalar (optional, ~30 min).** Track `(m, v, t)` server-side; step = `lr · m̂ / (√v̂ + ε)`. Closes the MEAZO-faithfulness gap vs the AIMD heuristic we shipped in Phase 39. Compare against symmetric AIMD as a final empirical sanity check before Phase 40.

3. **Phase 38b — multi-seed + multi-P sweep.** Phase 38/39's numbers are single-sample. For paper: ≥5 seeds per cell, P sweep {25K, 50K, 100K, 200K}. Run against deployed Worker (wrangler dev R>~95 is unstable). ~1 day.

4. **Phase 41 — VerifBFL zk-SNARK (arXiv:2501.04319).** Replace post-apply byzantine check with cryptographic guarantee that worker computed `scalar_g` honestly on committed data. 0.6 s on-chain verify, 81 s proof gen. Positions postnet against Gensyn / Prime Intellect's "verifiable training" pitch but for a SINGLE SCALAR rather than full gradient. ~1 week. The "splashy security" upgrade.

5. **Persistent DO state** (still deferred). All seven tournament DOs are in-memory. Declare `state.storage` schema, restore on construct, write on advance. ~30 min per DO. Becomes more important after deploying to prod.

**Strategic context from the May 2026 paper scans + NTK-Mirror find:**
- INTELLECT-3 (Prime Intellect, Nov 2025) went centralized — abandoned the decentralized story. "Anyone-can-join" frontier in mid-2026 = Nous DisTrO + Pluralis + postnet-cf + NTK-Mirror (the latter is single-machine but the gate parameterization is FL-shaped).
- NTK-Mirror is brand new (May 23, 2026) but going viral fast. Combining it with postnet-cf + neuropulse is a "three-MIT-projects compose into one Cloudflare Worker that trains real LLM behavior across browser tabs" story — genuinely novel positioning.

**The "obvious next move" if you say "go":** Phase 40 next-4-b session 4 — actually run the demo end-to-end in a real browser, then ship to prod. Session 3 wrote the browser code (`public/ntk.html` + `public/ntk-worker.js`) but couldn't validate it in-session (no claude-in-chrome).

**Phase 40 next-4-b session 4 plan:**
1. **User validation:** terminal 1 = `npm run dev`; terminal 2 = `cd ~/postnet-cf-onnx && npx http-server -p 8788 --cors -c-1 .`; open http://localhost:8787/ntk.html. Click Join, watch progress (download → ORT session → first forward → SPSA loop). Confirm loss descends.
2. **HF Hub upload of the 866 MB int8 ONNX** so prod doesn't depend on a sibling local server. Instructions in `docs/PHASE_40_NEXT4B_QWEN_ONNX.md`. After upload, flip `ONNX_URL` default in `public/ntk-worker.js`.
3. **Bump TARGET_PROPOSALS = 2** in `src/tournament-ntk.ts` once we know two browser tabs can run concurrently.
4. **WebGPU EP** (1-line change in worker; ~3-5× speedup if compatible).
5. **Longer empirical run** (R=200+) + writeup similar to Phase 37 crossover or Phase 39 sym-AIMD.

**Phase 40 next-4-b session 3 deliverables (DONE):**
- `public/ntk-worker.js` (~580 LOC) — ESM, onnxruntime-web from CDN, OPFS-cached 866 MB ONNX, SPSA loop mirroring Python verifier 1:1, baked tokenized math corpus.
- `public/ntk.html` — demo page paralleling head.html / lm.html with download-progress display.
- ONNX moved out of `public/data/` (wrangler dev rejects assets > 25 MiB) into `~/postnet-cf-onnx/`; documented sibling-http-server pattern + HF Hub upload path for prod.
- All Phase 40 next-4-b session 3 caveats + per-round wall-time estimates in `docs/PHASE_40_NEXT4B_QWEN_ONNX.md`.

**Phase 40 next-4-b session 2 deliverables (DONE):**
- `scripts/quantize-qwen-onnx.py` — int8 dynamic quantization via `onnxruntime.quantization.quantize_dynamic`; per-channel MatMul + Gemm; validated by forward-comparing logits to fp32 (top-1 token MUST match).
- Empirical: 866 MB int8 from 1.8 GB fp32 (47% reduction), forward 1166 ms vs 4366 ms (3.7× faster), top-1 token = 220 (' ') matches both backends.
- int4 path documented but not built (needs onnxruntime-genai model builder + ONNX surgery to inject gates).

**Phase 40 next-4-b session 1 deliverables (DONE):**
- `scripts/export-qwen-with-gates.py` — torch.onnx.export wrapper with forward hooks; produces ONNX whose inputs include `gate_mults: [24, 896] float32`
- `~/ntkmirror/.venv` has `onnx`, `onnxruntime`, `onnxscript`, `onnxruntime-genai` installed

**Phase 40 next-4-b session 2 deliverables (DONE):**
- `scripts/quantize-qwen-onnx.py` — int8 dynamic quantization via `onnxruntime.quantization.quantize_dynamic`; per-channel MatMul + Gemm; validated by forward-comparing logits to fp32 (top-1 token MUST match).
- Empirical: 866 MB int8 from 1.8 GB fp32 (47% reduction), forward 1166 ms vs 4366 ms (3.7× faster), top-1 token = 220 (' ') matches both backends.
- int4 path documented but not built (needs onnxruntime-genai model builder + ONNX surgery to inject gates).

**Phase 40 next-4-b session 1 deliverables (DONE):**
- `scripts/export-qwen-with-gates.py` — torch.onnx.export wrapper with forward hooks; produces ONNX whose inputs include `gate_mults: [24, 896] float32`
- `~/ntkmirror/.venv` has `onnx`, `onnxruntime`, `onnxscript`, `onnxruntime-genai` installed

**Phase 40 next-4-a deliverables (DONE):**
- `public/head-spsa-worker.js` — browser SPSA worker over the head-classifier MLP
- `public/head.html` — demo page paralleling lm.html / spsa-lm.html
- Open URL → click Join → federated-train AG News classifier across browser tabs of strangers

**Phase 40 next-3 deliverables (DONE):**
- `src/tournament-ntk.ts` — +124 LOC: lastLoss / pendingAudit / sym-AIMD / workerStats / quarantine
- `scripts/ntk-verifier.py` — +30 LOC: audit posting, server-η sync, per-round η/grow/shr logging
- `docs/PHASE_40_NEXT3_LOSS_ORACLE.md` — empirical writeup of the η-drift result
- Wire format gained one optional float: `audit_loss_before`

**Phase 40 next-2 deliverables (DONE):**
- `src/tournament-ntk.ts` — federated DO (K=5000 gates as trainable surface)
- `src/worker.ts` + `wrangler.jsonc` — binding + migration v10
- `scripts/ntk-verifier.py` — Python verifier using `ntkmirror`'s `_SignedLogMaskModule` to inject gates into Qwen-0.5B forward
- `docs/PHASE_40_NEXT2_NTK_DO.md` — empirical writeup of the R=30 run

**Still deferred (multi-phase backlog):**
- TARGET_PROPOSALS = 2 (currently 1 in `src/tournament-ntk.ts` for solo dev)
- An actual byzantine attacker test (logic is wired but no `--attack` flag yet)
- Longer empirical run (R = 200+) for paper-grade trajectory
- Persistent DO state (`state.storage`) across all 12 tournament DOs

**Phase 40 next-1 deliverables (DONE):**
- `scripts/extract-ntk-gates.py` — emits binary artifact (32 B header + K × 8 B body)
- `src/ntk-gate.ts` — `parseGateArtifact`, `buildGateIndex`, `applyGatesToLayer` / `applyGatesToStack`, `fnv1a64`
- `public/data/qwen05b-math-gates-k5000.bin` — 40 032 B (5000 gates × Qwen-0.5B)
- `scripts/test-ntk-gate-parse.mjs` — 16 checks, all pass; Python ↔ TS round-trip byte-verified

**Phase 40 scoping notes (carried from prior session):**
- ntkmirror cloned at `~/ntkmirror`, venv at `~/ntkmirror/.venv` (Python 3.14, torch 2.12, transformers 5.9).
- Mac flag REQUIRED: `--dtype fp32`. Defaults NaN at step 8 on MPS.
- Qwen2.5-0.5B-Instruct: 24 layers × 896 hidden = 21 504 candidate gates; we picked 5 000 (~23 %).
- Per-layer histogram (locked in): heavy on early layers (264-331/each), light in middle (68-110), spike on final layer (248) — classic NTK U-shape.
- Tractable browser compute per the design table: 5-20 min per 100 rounds via Transformers.js, 2-8 min via neuropulse WGSL.

**Quick orientation pointers:**

- `README.md` — landing page + 37-row phase table
- `docs/PHASE_37_SCALING.md` — scaling crossover empirical (this is the "why SPSA matters" doc)
- `docs/PROTOCOL.md` — wire-format spec (v0.5; needs bump to v0.6 to spec SPSA proposal/applied shapes)
- `docs/EMPIRICAL_STUDY.md` — multi-seed numbers (vanilla / sharded / byzantine, attacker-count sweep)
- `docs/PAPER_DRAFT.md` — 8-section arXiv-style writeup (predates Phase 36; needs an SPSA section before submission)
- `docs/OPEN_QUESTIONS.md` — what's not addressed yet (incl. the DKIM / Python / long-async extensions that are out of scope without the email v1 bridge)
- `scripts/empirical-study.mjs` — main driver for flip-and-accept comparisons
- `scripts/spsa-verifier.mjs` / `scripts/spsa-big-verifier.mjs` — SPSA verifiers
- `~/neuropulse/` — Phi-3 WebGPU engine (live demo of forward pass); the integration target for Phase 38

**Verify-by-reading rules:** when reviewing changes to `src/*.ts`, prefer reading over running smoke tests — `wrangler dev` is unstable under multi-verifier load (documented in Phase 19; reconfirmed Phase 37 ECONNRESET at R~1050). Production-deployed Workers handle the load fine.
