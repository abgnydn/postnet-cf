# ~/postnet-cf — CLAUDE.md

postnet-cf: federated tournament protocol on Cloudflare Workers. 34 phases shipped + deployed 2026-05-26; phases 35–37 (WebGPU substrate, SPSA fusion, scaling crossover) added 2026-05-26/27 — see "Resume here" for state.

## 🎯 Resume here (on "continue")

_Updated: 2026-07-13 (Phase 40 next-7(a) code shipped: worker default → HF Hub + scripts/upload-onnx-hf.py; one manual upload left)_

**Where we are.** Phase 40-4b-s6 closed the architecture limit from session 5b's live byzantine test. Three small fixes on `src/tournament-ntk.ts` + one on `public/ntk-worker.js`:
1. `pendingAudit` → `pendingAudits[]` queue (cap 64) with **no-self-audit rule** — incoming `audit_loss_before` from worker X can only close earliest pending entry whose `winnerId !== X`. Attackers can no longer close their own wins.
2. Reject `audit_loss_before ≤ 0` — real LM cross-entropy is positive; a zero sentinel was previously corrupting server's `lastLoss` baseline AND closing OTHER attackers' pending audits in a Sybil amplifier.
3. Magnitude-lie test added alongside Phase 39's inversion test: `claimedΔ < realΔ − MAGNITUDE_LIE_THRESH (= 0.5)` flags exaggerated claims missed by inversion when random perturbation happens to lower loss.
Two-tab live Chrome verified: cross-audit ledger grows correctly under attacker spam, attacker can't self-close, magnitude check would catch any audit it lands. Full writeup in `docs/PHASE_40_NEXT4B_QWEN_ONNX.md` "Session 6".

**Phase 40 next-6 deliverables (DONE):**
- `src/tournament-ntk.ts` — `pendingAudits[]` queue (cap 64) + no-self-audit rule + audit_loss_before > 0 guard + magnitude-lie fraud test (`MAGNITUDE_LIE_THRESH = 0.5`).
- `public/ntk-worker.js` — attacker no longer sends `audit_loss_before` (was 0; now `undefined` → JSON drops the field).
- `docs/PHASE_40_NEXT4B_QWEN_ONNX.md` "Session 6" — full writeup including what's NOT fixed (multi-tab coordinated Sybil → deferred to a future cryptographic-commitment phase).
- README.md — `40-4b-s6` row added.

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

**Phase 40 next-6e deliverable (DONE — empirical):** `docs/PHASE_40_NEXT6_EMPIRICAL.md` — python honest (Qwen on MPS, scripts/ntk-verifier.py) + browser attacker (`?attack=1`). Attacker quarantined at 10 audits (~150 s after honest came online); 100% fraud detection rate via magnitude-lie test. Post-quarantine loss 1.7632 → 1.7570 over ~120 honest rounds (server R=183); η climbed 1.0e-3 → 2.8e-3 monotonically (21 grow / 0 shrink). l2_norm of the K=5000 gate vector grew 0 → 0.634. README row `40-4b-s6e` added.

**Phase 40 next-6e-multi deliverable (DONE — multi-seed):** Same setup ported to Kaggle T4 GPU against the **deployed Worker** (`postnet-cf.abgunaydin94.workers.dev`). 5 seeds × R=100, python honest + python attacker thread. **5/5 quarantine.** Attacker fraud rate **1.000 ± 0.000** — every audit in every seed flagged. Final loss **1.762674 ± 0.000294** (σ below per-round Δloss). η-adaptation cadence varies more (grow 5.0 ± 2.5) but loss outcome is invariant to seed. Surfaced and fixed along the way: pip's UA → CF 403 (verifier + notebook now send a Chrome UA), Colab CPU-only fallback hangs Qwen forward (Kaggle T4 path documented). Notebooks: `notebooks/phase40_next6e_sweep.ipynb` (Colab), `notebooks/phase40_next6e_sweep_kaggle.ipynb` (Kaggle, the one used).

**Phase 40 next-7 (a) — HF Hub hosting (SHIPPED + LIVE 2026-07-13):**
- `public/ntk-worker.js` — `ONNX_URL` now resolves `?onnx=` → localhost `:8788`
  (dev) → **HuggingFace Hub `HF_ONNX_URL`** (prod default). Deployed demo no
  longer needs a local sibling http-server; localhost still auto-uses `:8788`.
- `scripts/upload-onnx-hf.py` — self-contained uv script (PEP 723; uses the
  cached HF token). Creates `abgunaydin/postnet-qwen05b-with-gates` + a model
  card (Apache-2.0 / NTK-Mirror attribution) and publishes the artifact at
  exactly the URL the worker expects.
- **Artifact rebuilt on this machine** (the file wasn't here; original was on the
  s5 machine). Full pipeline redone with a fresh uv env `~/postnet-cf-onnx/.venv`
  (optimum 2.1 + optimum-onnx 0.1 + transformers 4.57 + onnx 1.22 + ort 1.27):
  export (opset 17, dynamic batch) → `inject-gates-onnx.py` (logits diff 0.0 at
  gate=ones) → int8 quantize → **~906 MB single file** (863.9 MiB; the s5 toolchain
  gave 994 MB — same valid artifact, lighter export). Worker-faithful forward check
  (batch=4 baked tokens, all 4 inputs, gate=ones CE loss 4.34 ≈ s5's 4.1) PASSED.
- **Uploaded + verified LIVE**: `content-length 905846552`, `access-control-allow-origin: *`,
  `accept-ranges: bytes` on the CDN 200. Xet deduped vs base Qwen → only ~400 MB
  transferred. https://huggingface.co/abgunaydin/postnet-qwen05b-with-gates
- **REMAINING to light up the *deployed* demo:** the HF-default code is in the
  working tree but NOT yet committed/deployed — the live worker at
  `postnet-cf.abgunaydin94.workers.dev` still serves the old `:8788` default.
  Commit the next-7 changes + `npx wrangler deploy` to make the public demo load
  the HF file with zero setup.

**Still on deck after (a) deploys:**
- (b) **Phase 41 — VerifBFL zk-SNARK** for cryptographic commitment to data shard. Replaces "trust the audit" with "verify the proof." Sybil-resistant by construction. ~1 week. The splashy paper-grade upgrade.
- (c) **Multi-seed sweep** — largely covered by `40-4b-s6e-multi` (5 seeds × R=100 on Kaggle T4). Extend to R=200 for a publication-quality fraud-detection table if a paper needs it. ~1 hr.

**Phase 40 next-5b deliverables (DONE):**
- `public/ntk-worker.js` — attack mode now defined + short-circuits ONNX/ORT/snapshot loads; WebGPU EP attempt with WASM fallback (URL param `?backend=wasm|webgpu` overrides); `Response.bytes()` for single-allocation model fetch.
- `public/ntk.html` — `#attack` checkbox no longer disabled.
- `src/tournament-ntk.ts` — `TARGET_PROPOSALS = 2`.
- `docs/PHASE_40_NEXT4B_QWEN_ONNX.md` "Session 5b" — full writeup of the live byzantine test + the architecture limit + the three fixes for a future phase.

**Still queued from the original next-5 plan:**
- HF Hub upload of the 994 MB int8 ONNX (instructions in PHASE_40_NEXT4B_QWEN_ONNX.md); flip `ONNX_URL` default in worker.
- Longer empirical run (R=200+) for paper-grade trajectory data.

**Phase 40 next-4-b session 5 deliverables (DONE — end-to-end works):**
- Pipeline: `optimum-cli` (export) → `scripts/inject-gates-onnx.py` (Mul surgery) → `scripts/quantize-qwen-onnx.py` (int8 single file). Output: 994 MB single-file ONNX, ORT-web compatible.
- `public/ntk-worker.js`: updated to fetch the int8 ONNX, send `position_ids` (optimum-cli's 4th expected input), OPFS-cache with a 1.5 GB safety margin (Chrome's `blob.arrayBuffer()` rejects ≥2 GB; the 994 MB int8 file is the sweet spot).
- Live R=4 in Chrome: loss 4.10 → 3.85, η grew 1.0e-3 → 1.1e-3 (one sym-AIMD grow event), 3/4 proposals accepted by tournament.
- `docs/PHASE_40_NEXT4B_QWEN_ONNX.md` "Session 5" section: full pipeline reproducer + diagnostics for the three subtle issues we hit (optimum-onnx split package, external-data sidecar handling, ArrayBuffer max).

**Phase 40 next-4-b session 4 findings (DONE — negative result):**
- Architecture confirmed end-to-end: ONNX downloads, OPFS caches, ORT-web initializes session against our model (correct input/output names).
- But: `session.run()` aborts with raw WASM trap on first call, identical across WASM + WebGPU EPs, identical across fp32 + int8, identical at any batch×seq size.
- Diagnosed by loading Xenova/all-MiniLM-L6-v2 (21.9 MB int8 ONNX) in the same environment — works perfectly. So ORT-web is fine; our torch-produced ONNX has an op-coverage or graph-shape issue ORT-web doesn't handle.
- Tried `torch.onnx.export(..., dynamo=False)` (legacy tracer): produces a 1.4 MB ONNX with externalized initializers referring to FILES THAT WERE NEVER WRITTEN (broken artifact, ~58% of weights missing).
- Conclusion: `torch.onnx.export` is the wrong tool for ORT-web compatible Qwen ONNX. Need optimum-cli (the proven Xenova path) + post-hoc graph surgery to inject our 24 Mul nodes.

**Phase 40 next-4-b session 3 deliverables (DONE):**
- `public/ntk-worker.js` (~580 LOC) — structurally correct; will work once session 5 ships the right ONNX.
- `public/ntk.html` — demo page; works (loads worker, shows download progress, errors on forward as expected given session-4 finding).
- ONNX moved to `~/postnet-cf-onnx/`; sibling http-server pattern documented.

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
