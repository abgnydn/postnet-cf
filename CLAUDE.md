# ~/postnet-cf — CLAUDE.md

postnet-cf: federated tournament protocol on Cloudflare Workers. 34 phases shipped + deployed 2026-05-26; phases 35–37 (WebGPU substrate, SPSA fusion, scaling crossover) added 2026-05-26/27 — see "Resume here" for state.

## 🎯 Resume here (on "continue")

_Updated: 2026-05-28 (Phase 39 shipped; Phase 40 plan revised to NTK-Mirror integration)_

**Where we are.** Phase 39 shipped: adaptive η on the SPSA tournament. Tested two rules on the Phase 38 head-classifier (P=49 796). Asymmetric AIMD (×1.05 / ×0.7) collapsed η 16× and underperformed fixed-η. **Symmetric AIMD (×1.05 / ×1/1.05) won decisively at R=90: loss 1.40→1.12 (vs fixed-η 1.40→1.30 at R=100, i.e. 1.84× more descent), accuracy 32% → 56% (vs fixed-η 40%, +16 pp).** MEAZO's 2026 claim — that a single global scalar η matters more than per-param adaptivity — is supported at our scale. Wire format unchanged (η rides /tick responses). See `docs/PHASE_39_ADAPTIVE_ETA.md`.

**Files staged but uncommitted (as of 2026-05-28):**

- `src/tournament-spsa-lm.ts` + `src/tournament-spsa-lm-big.ts` — SPSA DO (P=2 379 and P=31 707) [Phase 36-37]
- `src/tournament-lm-big.ts` — flip-and-accept at the big scale (control) [Phase 37]
- `src/tournament-head-flip.ts` + `src/tournament-head-spsa.ts` + `src/head-model.ts` — head-classifier DOs + shared model [Phase 38]
- `src/tournament-head-spsa-adaptive.ts` + `scripts/head-spsa-adaptive-verifier.mjs` + `docs/PHASE_39_ADAPTIVE_ETA.md` — adaptive-η variant [Phase 39]
- `src/worker.ts` + `wrangler.jsonc` — bindings + migrations (now 7 tournament DOs total)
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

**The "obvious next move" if you say "go":** Phase 40 (NTK-Mirror integration). Phase 39b (Adam-on-scalar) is a quick optional sanity check that can be done as a warm-up if you want a half-day prelude.

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
